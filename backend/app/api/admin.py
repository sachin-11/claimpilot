"""Admin only: audit-log search, LLM spend and user management."""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import CurrentUser, require_admin
from app.core.db import get_session
from app.models import ActorType, AuditLog, User, UserRole
from app.observability.usage import IST, read_spend, today
from app.repositories.audit import AuditLogRepository

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


class AuditLogOut(BaseModel):
    id: UUID
    created_at: datetime
    actor_type: str
    actor_id: str | None
    action: str
    entity_type: str
    entity_id: str
    details: dict[str, Any]
    request_id: str | None


@router.get("/audit-logs")
async def audit_logs(
    entity_type: str | None = None,
    entity_id: str | None = None,
    actor_id: str | None = None,
    action: str | None = None,
    since: datetime | None = Query(None, description="created_at >= (ISO 8601)"),
    until: datetime | None = Query(None, description="created_at < (ISO 8601)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    conditions = []
    if entity_type:
        conditions.append(AuditLog.entity_type == entity_type)
    if entity_id:
        conditions.append(AuditLog.entity_id == entity_id)
    if actor_id:
        conditions.append(AuditLog.actor_id == actor_id)
    if action:
        conditions.append(AuditLog.action == action)
    if since:
        conditions.append(AuditLog.created_at >= since)
    if until:
        conditions.append(AuditLog.created_at < until)
    total = (
        await session.execute(select(func.count()).select_from(AuditLog).where(*conditions))
    ).scalar_one()
    rows = (
        (
            await session.execute(
                select(AuditLog)
                .where(*conditions)
                .order_by(AuditLog.created_at.desc(), AuditLog.id)
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    return {
        "total": total,
        "items": [
            AuditLogOut(
                id=r.id,
                created_at=r.created_at,
                actor_type=r.actor_type.value,
                actor_id=r.actor_id,
                action=r.action,
                entity_type=r.entity_type,
                entity_id=r.entity_id,
                details=dict(r.details or {}),
                request_id=r.request_id,
            )
            for r in rows
        ],
    }


@router.get("/usage")
async def usage(request: Request, days: int = Query(7, ge=1, le=10)) -> dict[str, Any]:
    """Estimated LLM spend (INR) today against the daily budget, and the last few days."""
    settings = request.app.state.settings
    redis = request.app.state.redis
    now = datetime.now(IST)
    history = [
        await read_spend(redis, today(now - timedelta(days=offset))) for offset in range(days)
    ]
    current = history[0]
    budget = settings.daily_llm_budget_inr
    return {
        "currency": "INR",
        "usd_inr_rate": settings.usd_inr_rate,
        "today": {
            **current,
            "budget_inr": budget,
            "remaining_inr": round(max(budget - current["inr"], 0), 4) if budget > 0 else None,
            "budget_used_percent": round(100 * current["inr"] / budget, 1) if budget > 0 else None,
        },
        "history": [
            {k: h[k] for k in ("day", "inr", "calls", "tokens_in", "tokens_out")} for h in history
        ],
        "note": "estimates from list prices (app/observability/pricing.yaml), not a bill",
    }


class UserOut(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: str
    is_active: bool


def _user(u: User) -> UserOut:
    return UserOut(
        id=u.id, email=u.email, full_name=u.full_name, role=u.role.value, is_active=u.is_active
    )


class UserCreate(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=255)
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=12, max_length=128)
    role: UserRole = UserRole.OFFICER


class UserUpdate(BaseModel):
    role: UserRole | None = None
    is_active: bool | None = None


@router.get("/users")
async def list_users(session: AsyncSession = Depends(get_session)) -> list[UserOut]:
    rows = (await session.execute(select(User).order_by(User.email))).scalars().all()
    return [_user(u) for u in rows]


@router.post("/users", status_code=201)
async def create_user(
    body: UserCreate,
    request: Request,
    admin: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    user = User(
        email=body.email.lower(),
        full_name=body.full_name,
        role=body.role,
        password_hash=bcrypt.hashpw(body.password.encode()[:72], bcrypt.gensalt()).decode(),
    )
    session.add(user)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(409, "a user with this e-mail already exists") from exc
    await AuditLogRepository(session).log(
        actor_type=ActorType.USER,
        actor_id=str(admin.id),
        action="user_created",
        entity_type="user",
        entity_id=user.id,
        details={"role": user.role.value},
        request_id=getattr(request.state, "request_id", None),
    )
    await session.commit()
    return _user(user)


@router.patch("/users/{user_id}")
async def update_user(
    user_id: UUID,
    body: UserUpdate,
    request: Request,
    admin: CurrentUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> UserOut:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    if user.id == admin.id and (body.is_active is False or body.role == UserRole.OFFICER):
        raise HTTPException(409, "you cannot disable or demote your own account")
    changes: dict[str, Any] = {}
    if body.role is not None and body.role != user.role:
        user.role = body.role
        changes["role"] = body.role.value
    if body.is_active is not None and body.is_active != user.is_active:
        user.is_active = body.is_active
        changes["is_active"] = body.is_active
    if changes:
        await AuditLogRepository(session).log(
            actor_type=ActorType.USER,
            actor_id=str(admin.id),
            action="user_updated",
            entity_type="user",
            entity_id=user.id,
            details=changes,
            request_id=getattr(request.state, "request_id", None),
        )
    await session.commit()
    return _user(user)
