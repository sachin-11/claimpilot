"""Login, refresh (with rotation and reuse detection) and logout, plus the login lockout.

The caller owns the transaction (commit after the call), like every repository here."""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import (
    create_access_token,
    hash_refresh_token,
    new_refresh_token,
    verify_password,
)
from app.core.config import Settings
from app.models import ActorType, RefreshToken, User
from app.repositories.audit import AuditLogRepository


class AuthError(Exception):
    """Wrong credentials or an unusable token (HTTP 401); never says which part was wrong."""


class LoginLockedError(Exception):
    """Too many failed attempts for this e-mail and address (HTTP 429)."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("too many failed login attempts")
        self.retry_after = retry_after


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    expires_at: datetime
    user: User

    token_type: str = "bearer"


def _attempt_key(email: str, ip: str) -> str:
    digest = hashlib.sha256(email.strip().lower().encode()).hexdigest()[:32]
    return f"login_fail:{digest}:{ip}"


async def check_not_locked(redis: Any, settings: Settings, email: str, ip: str) -> None:
    count = await redis.get(_attempt_key(email, ip))
    if count is not None and int(count) >= settings.login_max_failures:
        ttl = await redis.ttl(_attempt_key(email, ip))
        raise LoginLockedError(max(int(ttl), 1))


async def record_failure(redis: Any, settings: Settings, email: str, ip: str) -> None:
    key = _attempt_key(email, ip)
    count = await redis.incr(key)
    if count == 1:  # the window starts at the first failure
        await redis.expire(key, settings.login_window_minutes * 60)


async def clear_failures(redis: Any, email: str, ip: str) -> None:
    await redis.delete(_attempt_key(email, ip))


async def _issue(
    session: AsyncSession, settings: Settings, user: User, *, ip: str | None, agent: str | None
) -> tuple[TokenPair, RefreshToken]:
    access, expires = create_access_token(
        settings, user_id=user.id, role=user.role, email=user.email
    )
    refresh = new_refresh_token()
    row = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(refresh),
        expires_at=datetime.now(UTC) + timedelta(days=settings.refresh_token_days),
        client_ip=ip,
        user_agent=(agent or "")[:255] or None,
    )
    session.add(row)
    await session.flush()
    return TokenPair(access, refresh, expires, user), row


async def login(
    session: AsyncSession,
    redis: Any,
    settings: Settings,
    *,
    email: str,
    password: str,
    ip: str,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> TokenPair:
    await check_not_locked(redis, settings, email, ip)
    user = (
        await session.execute(select(User).where(User.email == email.strip().lower()))
    ).scalar_one_or_none()
    ok = verify_password(password, user.password_hash if user else None)
    audit = AuditLogRepository(session)
    if not ok or user is None or not user.is_active:
        await record_failure(redis, settings, email, ip)
        await audit.log(
            actor_type=ActorType.SYSTEM,
            action="login_failed",
            entity_type="user",
            entity_id=str(user.id) if user else uuid.UUID(int=0),
            details={"ip": ip},
            request_id=request_id,
        )
        raise AuthError("invalid e-mail or password")
    await clear_failures(redis, email, ip)
    pair, _ = await _issue(session, settings, user, ip=ip, agent=user_agent)
    await audit.log(
        actor_type=ActorType.USER,
        actor_id=str(user.id),
        action="login",
        entity_type="user",
        entity_id=user.id,
        details={"ip": ip},
        request_id=request_id,
    )
    return pair


async def refresh(
    session: AsyncSession,
    settings: Settings,
    *,
    refresh_token: str,
    ip: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
) -> TokenPair:
    """Rotate: the presented token is revoked and replaced. A token that was already rotated is
    evidence of theft, so every refresh token of that user is revoked."""
    now = datetime.now(UTC)
    row = (
        await session.execute(
            select(RefreshToken)
            .where(RefreshToken.token_hash == hash_refresh_token(refresh_token))
            .with_for_update()
        )
    ).scalar_one_or_none()
    audit = AuditLogRepository(session)
    if row is None:
        raise AuthError("invalid refresh token")
    if row.revoked_at is not None:
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == row.user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        await audit.log(
            actor_type=ActorType.SYSTEM,
            action="refresh_token_reuse",
            entity_type="user",
            entity_id=row.user_id,
            details={"ip": ip},
            request_id=request_id,
        )
        await session.commit()  # the revocation must survive the 401 below
        raise AuthError("refresh token already used")
    if row.expires_at <= now:
        raise AuthError("refresh token expired")
    user = (await session.execute(select(User).where(User.id == row.user_id))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise AuthError("account disabled")
    pair, new_row = await _issue(session, settings, user, ip=ip, agent=user_agent)
    row.revoked_at = now
    row.replaced_by = new_row.id
    await session.flush()
    return pair


async def logout(
    session: AsyncSession,
    *,
    refresh_token: str,
    everywhere: bool = False,
    request_id: str | None = None,
) -> None:
    """Revoke this refresh token (or all of the user's). Unknown tokens are ignored (idempotent)."""
    now = datetime.now(UTC)
    row = (
        await session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(refresh_token))
        )
    ).scalar_one_or_none()
    if row is None:
        return
    query = update(RefreshToken).where(RefreshToken.revoked_at.is_(None))
    query = (
        query.where(RefreshToken.user_id == row.user_id)
        if everywhere
        else query.where(RefreshToken.id == row.id)
    )
    await session.execute(query.values(revoked_at=now))
    await AuditLogRepository(session).log(
        actor_type=ActorType.USER,
        actor_id=str(row.user_id),
        action="logout_all" if everywhere else "logout",
        entity_type="user",
        entity_id=row.user_id,
        request_id=request_id,
    )
