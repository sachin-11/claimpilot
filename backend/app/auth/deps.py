"""FastAPI dependencies: who is calling (JWT), what they may do (roles), and the LLM guards."""

import logging
import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.limits import client_ip, enforce, ensure_budget
from app.auth.security import TokenError, decode_access_token
from app.core.db import get_session
from app.models import User, UserRole

logger = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> CurrentUser:
    """The authenticated, active user. The role is read from the database, not from the token, so a
    demotion or a disabled account applies at once instead of when the access token expires."""
    settings = request.app.state.settings
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("authentication required")
    try:
        claims = decode_access_token(settings, credentials.credentials)
    except TokenError as exc:
        raise _unauthorized(str(exc)) from exc
    user = await session.get(User, claims.user_id)
    if user is None or not user.is_active:
        raise _unauthorized("account not found or disabled")
    current = CurrentUser(user.id, user.email, user.full_name, user.role)
    request.state.user = current
    await enforce(request, settings, "user", str(user.id), settings.rate_limit_user_per_minute)
    return current


def require_roles(*roles: UserRole):  # type: ignore[no-untyped-def]
    allowed = frozenset(roles)

    async def dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed:
            logger.warning("forbidden", extra={"user_id": str(user.id), "role": user.role.value})
            raise HTTPException(403, "your role is not allowed to do this")
        return user

    return dependency


# officer endpoints are open to admins too ("admin: everything")
require_officer = require_roles(UserRole.OFFICER, UserRole.ADMIN)
require_admin = require_roles(UserRole.ADMIN)


async def llm_guard(request: Request, user: CurrentUser = Depends(require_officer)) -> CurrentUser:
    """For endpoints that start LLM work: stricter per-user / per-IP limits and the daily budget."""
    settings = request.app.state.settings
    await enforce(
        request, settings, "llm-user", str(user.id), settings.rate_limit_llm_user_per_minute
    )
    await enforce(
        request, settings, "llm-ip", client_ip(request), settings.rate_limit_llm_ip_per_minute
    )
    await ensure_budget(request, settings)
    return user
