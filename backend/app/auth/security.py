"""Password check, access tokens (JWT) and refresh tokens. Pure functions: no database, no Redis."""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.core.config import Settings
from app.models.enums import UserRole

# A real bcrypt hash of a random password: checked when the e-mail is unknown, so a missing account
# costs the same time as a wrong password.
_DUMMY_HASH = bcrypt.hashpw(secrets.token_bytes(16), bcrypt.gensalt()).decode()


class TokenError(Exception):
    """The access token is missing, malformed, expired or not an access token."""


@dataclass(frozen=True)
class AccessClaims:
    user_id: uuid.UUID
    role: UserRole
    email: str
    expires_at: datetime


def verify_password(password: str, password_hash: str | None) -> bool:
    candidate = password.encode()[:72]  # bcrypt ignores everything past 72 bytes
    try:
        return bcrypt.checkpw(candidate, (password_hash or _DUMMY_HASH).encode()) and bool(
            password_hash
        )
    except ValueError:  # a malformed hash in the database is a failed login, not a crash
        return False


def create_access_token(
    settings: Settings,
    *,
    user_id: uuid.UUID,
    role: UserRole,
    email: str,
    now: datetime | None = None,
) -> tuple[str, datetime]:
    issued = now or datetime.now(UTC)
    expires = issued + timedelta(minutes=settings.access_token_minutes)
    payload = {
        "sub": str(user_id),
        "role": role.value,
        "email": email,
        "type": "access",
        "jti": uuid.uuid4().hex,
        "iat": int(issued.timestamp()),
        "exp": int(expires.timestamp()),
    }
    token = jwt.encode(
        payload, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )
    return token, expires


def decode_access_token(settings: Settings, token: str) -> AccessClaims:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],  # never trust the algorithm named in the token
            options={"require": ["exp", "sub", "iat"]},
        )
        if payload.get("type") != "access":
            raise TokenError("not an access token")
        return AccessClaims(
            user_id=uuid.UUID(payload["sub"]),
            role=UserRole(payload["role"]),
            email=str(payload.get("email", "")),
            expires_at=datetime.fromtimestamp(payload["exp"], UTC),
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("token expired") from exc
    except (jwt.PyJWTError, ValueError, KeyError) as exc:
        raise TokenError("invalid token") from exc


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """Only this hash is stored: a database leak does not leak usable refresh tokens."""
    return hashlib.sha256(token.encode()).hexdigest()
