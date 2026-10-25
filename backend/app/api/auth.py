"""POST /auth/login, /auth/refresh, /auth/logout and GET /auth/me.

The refresh token never appears in a response body: it travels as an httpOnly cookie
(SameSite=Strict, Secure in production), so a script in the page cannot read it. The access token is
returned in the body and kept in memory by the client. `/auth/refresh` and `/auth/logout` also
require an `X-Requested-With` header: a cross-site form post cannot add one, and a cross-site fetch
that does needs a CORS preflight the API refuses.
"""

import re
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import service
from app.auth.deps import CurrentUser, get_current_user
from app.auth.limits import client_ip
from app.core.config import Settings
from app.core.db import get_session
from app.core.errors import error_response

router = APIRouter(prefix="/auth", tags=["auth"])

_PREFIX_RE = re.compile(r"^/[A-Za-z0-9/_-]{0,64}$")


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=256)


class LogoutRequest(BaseModel):
    everywhere: bool = False  # revoke every refresh token of the user


class UserOut(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserOut


def cookie_path(request: Request, settings: Settings) -> str:
    """Where the refresh cookie is sent. Behind a proxy that mounts the API under a prefix (nginx /
    the Vite dev proxy send `X-Forwarded-Prefix: /api`) the browser sees `/api/auth/...`, so the
    cookie's path has to be `/api/auth`; reached directly it is `/auth`."""
    if settings.auth_cookie_path:
        return settings.auth_cookie_path
    prefix = request.headers.get("X-Forwarded-Prefix", "").rstrip("/")
    return (prefix if prefix and _PREFIX_RE.match(prefix) else "") + "/auth"


def set_refresh_cookie(
    response: Response, request: Request, settings: Settings, token: str
) -> None:
    response.set_cookie(
        settings.auth_cookie_name,
        token,
        max_age=settings.refresh_token_days * 86400,
        httponly=True,
        secure=settings.is_production,  # plain http on a developer machine
        samesite="strict",
        path=cookie_path(request, settings),
    )


def clear_refresh_cookie(response: Response, request: Request, settings: Settings) -> None:
    response.delete_cookie(
        settings.auth_cookie_name,
        path=cookie_path(request, settings),
        httponly=True,
        secure=settings.is_production,
        samesite="strict",
    )


def _tokens(pair: service.TokenPair) -> TokenResponse:
    return TokenResponse(
        access_token=pair.access_token,
        expires_at=pair.expires_at,
        user=UserOut(
            id=pair.user.id,
            email=pair.user.email,
            full_name=pair.user.full_name,
            role=pair.user.role.value,
        ),
    )


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def _require_ajax_header(request: Request) -> None:
    if not request.headers.get("X-Requested-With"):
        raise HTTPException(403, "missing X-Requested-With header")


@router.post("/login")
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    settings = request.app.state.settings
    try:
        pair = await service.login(
            session,
            request.app.state.redis,
            settings,
            email=body.email,
            password=body.password,
            ip=client_ip(request),
            user_agent=request.headers.get("User-Agent"),
            request_id=getattr(request.state, "request_id", None),
        )
    except service.LoginLockedError as exc:
        raise HTTPException(
            429,
            "too many failed login attempts; try again later",
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except service.AuthError as exc:
        await session.commit()  # keep the failed-login audit row
        raise _unauthorized(str(exc)) from exc
    await session.commit()
    set_refresh_cookie(response, request, settings, pair.refresh_token)
    return _tokens(pair)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request, response: Response, session: AsyncSession = Depends(get_session)
) -> Any:
    _require_ajax_header(request)
    settings = request.app.state.settings
    token = request.cookies.get(settings.auth_cookie_name)
    if not token:
        raise _unauthorized("no refresh token")
    try:
        pair = await service.refresh(
            session,
            settings,
            refresh_token=token,
            ip=client_ip(request),
            user_agent=request.headers.get("User-Agent"),
            request_id=getattr(request.state, "request_id", None),
        )
    except service.AuthError as exc:
        failure = error_response(
            request, 401, "unauthorized", str(exc), headers={"WWW-Authenticate": "Bearer"}
        )
        clear_refresh_cookie(failure, request, settings)  # a dead token is no use to the browser
        return failure
    await session.commit()
    set_refresh_cookie(response, request, settings, pair.refresh_token)
    return _tokens(pair)


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    body: LogoutRequest | None = None,
    session: AsyncSession = Depends(get_session),
) -> Response:
    _require_ajax_header(request)
    settings = request.app.state.settings
    token = request.cookies.get(settings.auth_cookie_name)
    if token:
        await service.logout(
            session,
            refresh_token=token,
            everywhere=bool(body and body.everywhere),
            request_id=getattr(request.state, "request_id", None),
        )
        await session.commit()
    response = Response(status_code=204)
    clear_refresh_cookie(response, request, settings)
    return response


@router.get("/me")
async def me(user: CurrentUser = Depends(get_current_user)) -> UserOut:
    return UserOut(id=user.id, email=user.email, full_name=user.full_name, role=user.role.value)
