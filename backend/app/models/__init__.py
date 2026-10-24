"""Importing this package registers every model on Base.metadata (Alembic relies on it)."""

from app.models.audit import AuditLog
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
    UserRole,
)
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
    "Policy",
    "PolicyChunk",
    "User",
    "UserRole",
]
