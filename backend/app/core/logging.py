import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

# Set by the request middleware; every log line emitted while handling a request carries it.
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

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
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
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
