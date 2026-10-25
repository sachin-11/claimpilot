"""Regex-only PII redaction for log output (kept free of heavy imports on purpose: the logging
setup imports this on every start)."""

import logging
import re

# landlines and numbers not starting 6-9 occur in real data too: any standalone 10-digit run
# Not part of a longer number, date or amount ("12/03/2025", "1,23,456.00") but fine before / after
# ordinary punctuation: "Aadhaar 2345 6789 0123, PAN ..." must still be redacted.
_NOT_AFTER = r"(?<!\d)(?<!\d[/.,-])"
_NOT_BEFORE = r"(?!\d)(?![/.,-]\d)"
PHONE_RE = re.compile(_NOT_AFTER + r"(?:(?:\+?91)[\s-]?)?0?\d{5}[\s-]?\d{5}" + _NOT_BEFORE)
AADHAAR_RE = re.compile(_NOT_AFTER + r"[2-9]\d{3}\s?\d{4}\s?\d{4}" + _NOT_BEFORE)
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (EMAIL_RE, "[email]"),
    (AADHAAR_RE, "[aadhaar]"),
    (PHONE_RE, "[phone]"),
    (PAN_RE, "[pan]"),
]


def redact(text: str) -> str:
    """Replace e-mail / Aadhaar / phone / PAN patterns."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def redact_any(value: object) -> object:
    """redact() applied to strings inside nested dicts / lists / tuples."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_any(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return type(value)(redact_any(v) for v in value)
    return value


_STANDARD_ATTRIBUTES = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class PIIRedactingFilter(logging.Filter):
    """Redacts phone / e-mail / Aadhaar / PAN patterns in the message and in every `extra` field."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_ATTRIBUTES:
                setattr(record, key, redact_any(value))
        if record.exc_info and not record.exc_text:
            # format the traceback here so it can be redacted too (the exception message may
            # contain personal data), then drop exc_info so nobody formats it again unredacted
            record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True
