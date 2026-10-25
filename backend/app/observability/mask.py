"""Langfuse mask function: the Module 6 redaction runs on everything before it leaves the app."""

from typing import Any

from app.core.redaction import redact

# Dict keys whose value is personal data whatever it looks like (names have no regex).
SENSITIVE_KEYS = frozenset(
    {
        "patient_name", "insured_name", "full_name", "customer_name", "policyholder_name",
        "first_name", "last_name", "phone", "mobile", "contact", "email", "aadhaar", "aadhar",
        "pan", "address", "dob", "date_of_birth",
    }
)  # fmt: skip
REDACTED = "[redacted]"


def _mask(value: Any, key: str | None = None) -> Any:
    if key is not None and key.lower() in SENSITIVE_KEYS and isinstance(value, (str, int, float)):
        return REDACTED
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _mask(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_mask(v) for v in value)
    return value


def mask_trace_data(*, data: Any, **_: Any) -> Any:
    """Langfuse `mask=` hook (keyword-only `data`). Fails closed: if masking itself fails the
    value is replaced, never sent as it was."""
    try:
        return _mask(data)
    except Exception:  # noqa: BLE001
        return "[masking failed]"
