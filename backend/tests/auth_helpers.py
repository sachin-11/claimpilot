"""Test helpers for the authenticated API: skip the JWT check with a given user, or build real tokens."""

import uuid
from datetime import UTC, datetime

from fastapi import FastAPI

from app.auth.deps import CurrentUser, get_current_user
from app.auth.security import create_access_token
from app.core.config import Settings
from app.models import User, UserRole

FAKE_ADMIN = CurrentUser(
    uuid.UUID("00000000-0000-0000-0000-0000000000ad"),
    "admin@test.example",
    "Test Admin",
    UserRole.ADMIN,
)
FAKE_OFFICER = CurrentUser(
    uuid.UUID("00000000-0000-0000-0000-0000000000f1"),
    "officer@test.example",
    "Test Officer",
    UserRole.OFFICER,
)


def current_user_for(user: User) -> CurrentUser:
    return CurrentUser(user.id, user.email, user.full_name, user.role)


def bypass_auth(app: FastAPI, user: CurrentUser | None = None) -> CurrentUser:
    """Every request is made by `user` (an admin by default): for tests that are not about auth."""
    acting = user or FAKE_ADMIN

    async def override() -> CurrentUser:
        return acting

    app.dependency_overrides[get_current_user] = override
    return acting


def bearer(settings: Settings, user: User, now: datetime | None = None) -> dict[str, str]:
    token, _ = create_access_token(
        settings, user_id=user.id, role=user.role, email=user.email, now=now or datetime.now(UTC)
    )
    return {"Authorization": f"Bearer {token}"}
