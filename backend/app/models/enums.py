from enum import StrEnum


class ClaimStatus(StrEnum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    PARTIALLY_APPROVED = "partially_approved"
    REJECTED = "rejected"
    DOCUMENTS_REQUESTED = "documents_requested"
    FAILED = "failed"


class ClaimSource(StrEnum):
    SEED = "seed"
    UPLOAD = "upload"


class DocType(StrEnum):
    HOSPITAL_BILL = "hospital_bill"
    DISCHARGE_SUMMARY = "discharge_summary"
    CLAIM_FORM = "claim_form"
    OTHER = "other"


# Same values as data_gen.models.Decision (Module 1 golden data).
class DecisionType(StrEnum):
    APPROVE = "approve"
    PARTIAL_APPROVE = "partial_approve"
    REJECT = "reject"
    FLAG_FOR_REVIEW = "flag_for_review"
    REQUEST_DOCUMENTS = "request_documents"


class ApprovalAction(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


class UserRole(StrEnum):
    OFFICER = "officer"
    ADMIN = "admin"


class ActorType(StrEnum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class IngestionStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class NotificationChannel(StrEnum):
    EMAIL = "email"
    SMS = "sms"


class NotificationStatus(StrEnum):
    QUEUED = "queued"
    SENT_MOCK = "sent_mock"
