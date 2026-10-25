"""Prometheus scrape endpoint (not in the OpenAPI docs). Needs the METRICS_TOKEN bearer token
(for the Prometheus server) or an admin access token."""

import hmac

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import CurrentUser, get_current_user
from app.core.db import get_session
from app.observability.metrics import render_metrics

router = APIRouter(tags=["metrics"], include_in_schema=False)
_bearer = HTTPBearer(auto_error=False)


async def metrics_access(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> None:
    settings = request.app.state.settings
    token = settings.metrics_token.get_secret_value() if settings.metrics_token else ""
    presented = credentials.credentials if credentials else ""
    if token and presented and hmac.compare_digest(token.encode(), presented.encode()):
        return
    user: CurrentUser = await get_current_user(request, credentials, session)
    if not user.is_admin:
        raise HTTPException(403, "metrics are for admins and the metrics scraper")


@router.get("/metrics")
async def metrics(_: None = Depends(metrics_access)) -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
