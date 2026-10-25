import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from app.core.redaction import PIIRedactingFilter

# Set by the request middleware; every log line emitted while handling a request carries it.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
# Set while a traced block runs (see `bind_log_context`); added to every log line when present.
trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)
run_id_ctx: ContextVar[str | None] = ContextVar("run_id", default=None)
claim_number_ctx: ContextVar[str | None] = ContextVar("claim_number", default=None)


@contextmanager
def bind_log_context(
    *, trace_id: str | None = None, run_id: str | None = None, claim_number: str | None = None
) -> Iterator[None]:
    """trace_id / run_id / claim_number on every log line emitted inside the block (also by the
    tasks it starts, which copy the context)."""
    tokens = [
        (var, var.set(value))
        for var, value in (
            (trace_id_ctx, trace_id),
            (run_id_ctx, run_id),
            (claim_number_ctx, claim_number),
        )
        if value is not None
    ]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


# Attributes every LogRecord has. Anything else came from `extra={...}` and is emitted as-is.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        for key, var in (
            ("trace_id", trace_id_ctx),
            ("run_id", run_id_ctx),
            ("claim_number", claim_number_ctx),
        ):
            if (value := var.get()) is not None:
                payload[key] = value
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        elif record.exc_text:  # already formatted (and redacted) by PIIRedactingFilter
            payload["exc_info"] = record.exc_text
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(PIIRedactingFilter())  # no raw phone / e-mail / Aadhaar in any log line
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Route uvicorn through our JSON handler. Access lines are replaced by the timing middleware.
    for name in ("uvicorn", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers = []
        uv.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
