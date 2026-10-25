"""POST /auth/login, /auth/refresh, /auth/logout and GET /auth/me."""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import service
from app.auth.deps import CurrentUser, get_current_user
from app.auth.limits import client_ip
from app.core.db import get_session

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


class LogoutRequest(RefreshRequest):
    everywhere: bool = False  # revoke every refresh token of the user


class UserOut(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserOut


def _tokens(pair: service.TokenPair) -> TokenResponse:
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
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


@router.post("/login")
async def login(
    body: LoginRequest, request: Request, session: AsyncSession = Depends(get_session)
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
    return _tokens(pair)


@router.post("/refresh")
async def refresh(
    body: RefreshRequest, request: Request, session: AsyncSession = Depends(get_session)
) -> TokenResponse:
    try:
        pair = await service.refresh(
            session,
            request.app.state.settings,
            refresh_token=body.refresh_token,
            ip=client_ip(request),
            user_agent=request.headers.get("User-Agent"),
            request_id=getattr(request.state, "request_id", None),
        )
    except service.AuthError as exc:
        raise _unauthorized(str(exc)) from exc
    await session.commit()
    return _tokens(pair)


@router.post("/logout", status_code=204)
async def logout(
    body: LogoutRequest, request: Request, session: AsyncSession = Depends(get_session)
) -> Response:
    await service.logout(
        session,
        refresh_token=body.refresh_token,
        everywhere=body.everywhere,
        request_id=getattr(request.state, "request_id", None),
    )
    await session.commit()
    return Response(status_code=204)


@router.get("/me")
async def me(user: CurrentUser = Depends(get_current_user)) -> UserOut:
    return UserOut(id=user.id, email=user.email, full_name=user.full_name, role=user.role.value)
