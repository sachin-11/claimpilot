"""Importing this package registers every model on Base.metadata (Alembic relies on it)."""

from app.models.audit import AuditLog
from app.models.auth import RefreshToken
from app.models.claim import (
    Claim,
    ClaimApproval,
    ClaimDecision,
    ClaimDocument,
    ClaimExtraction,
)
from app.models.customer import Customer, CustomerPolicy
from app.models.enums import (
    ActorType,
    ApprovalAction,
    ClaimSource,
    ClaimStatus,
    DecisionType,
    DocType,
    IngestionStatus,
    NotificationChannel,
    NotificationStatus,
    UserRole,
)
from app.models.idempotency import ToolIdempotency
from app.models.ingestion import IngestionRun
from app.models.notification import Notification
from app.models.policy import Policy, PolicyChunk
from app.models.user import User

__all__ = [
    "ActorType",
    "ApprovalAction",
    "AuditLog",
    "Claim",
    "ClaimApproval",
    "ClaimDecision",
    "ClaimDocument",
    "ClaimExtraction",
    "ClaimSource",
    "ClaimStatus",
    "Customer",
    "CustomerPolicy",
    "DecisionType",
    "DocType",
    "IngestionRun",
    "IngestionStatus",
    "Notification",
    "NotificationChannel",
    "NotificationStatus",
    "Policy",
    "PolicyChunk",
    "RefreshToken",
    "ToolIdempotency",
    "User",
    "UserRole",
]
