"""Optional Langfuse tracing: one `claim_run` trace per agent run, `policy_query` per Q&A, spans for
the non-LLM steps. Without both Langfuse keys nothing is imported or sent and every helper here is a
no-op, so the app behaves exactly as before.

IDs: the trace id is derived from a seed (the run id for a claim run), so the API can return the
`X-Trace-ID` header before the run starts and a resume after the human approval joins the same
trace. Everything leaving the app goes through `mask_trace_data` (Module 6 redaction)."""

import hashlib
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, has_real_key
from app.observability.mask import mask_trace_data

logger = logging.getLogger(__name__)

_client: Any = None  # langfuse.Langfuse once init_tracing() succeeded
_callbacks_ctx: ContextVar[tuple[Any, ...]] = ContextVar("trace_callbacks", default=())


def current_callbacks() -> list[Any]:
    """LangChain callbacks of the active trace (empty when tracing is off). LLM calls inside a
    LangGraph run get them from the graph config; plain calls (RAG answer, extraction) pass
    `config={"callbacks": current_callbacks()}`."""
    return list(_callbacks_ctx.get())


def callback_config() -> dict[str, Any]:
    """`**callback_config()` for an `ainvoke`: the active trace's callbacks, or nothing at all when
    tracing is off (so the call is exactly what it was without tracing)."""
    callbacks = current_callbacks()
    return {"config": {"callbacks": callbacks}} if callbacks else {}


def tracing_enabled() -> bool:
    return _client is not None


def trace_id_for(seed: str) -> str:
    """32 hex characters (the Langfuse / OpenTelemetry trace id format) derived from `seed`."""
    return hashlib.sha256(seed.encode()).hexdigest()[:32]


def init_tracing(settings: Settings) -> bool:
    """Create the Langfuse client (with the PII mask). False = tracing stays off."""
    global _client
    if _client is not None:
        return True
    if not (
        has_real_key(settings.langfuse_public_key) and has_real_key(settings.langfuse_secret_key)
    ):
        logger.info("tracing_disabled", extra={"reason": "LANGFUSE keys are not set"})
        return False
    try:
        from langfuse import Langfuse  # noqa: PLC0415 - optional dependency path
        from langfuse.span_filter import is_langfuse_span  # noqa: PLC0415

        _client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),  # type: ignore[union-attr]
            secret_key=settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
            base_url=settings.langfuse_base_url or settings.langfuse_host or None,
            environment=settings.app_env,
            mask=mask_trace_data,
            # Only spans made through the Langfuse SDK (this module, the LangChain handler).
            # Third-party instrumentation (e.g. the Anthropic SDK's own spans) would bypass the
            # PII mask and count every LLM call twice.
            should_export_span=is_langfuse_span,
        )
    except Exception:  # noqa: BLE001 - tracing must never stop the app from starting
        logger.warning("tracing_init_failed", exc_info=True)
        _client = None
        return False
    logger.info("tracing_enabled", extra={"base_url": settings.langfuse_base_url or "default"})
    return True


def flush_tracing() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception:  # noqa: BLE001
            logger.warning("tracing_flush_failed", exc_info=True)


def shutdown_tracing() -> None:
    global _client
    if _client is not None:
        try:
            _client.shutdown()
        except Exception:  # noqa: BLE001
            logger.warning("tracing_shutdown_failed", exc_info=True)
        _client = None


@dataclass
class TraceHandle:
    """What a traced block hands back: ids, the LangChain callbacks to pass to LLM / graph calls,
    and `update()` for the root observation (output, metadata known only at the end)."""

    trace_id: str
    callbacks: list[Any] = field(default_factory=list)
    root: Any = None

    def update(self, **fields: Any) -> None:
        if self.root is not None:
            try:
                self.root.update(**fields)
            except Exception:  # noqa: BLE001
                logger.warning("trace_update_failed", exc_info=True)


@contextmanager
def run_trace(
    name: str,
    *,
    seed: str,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
    input: Any = None,  # noqa: A002
) -> Iterator[TraceHandle]:
    """A trace named `name` with a deterministic id from `seed`. Calling it again with the same seed
    adds to the same trace (a run resumed after the approval)."""
    trace_id = trace_id_for(seed)
    if _client is None:
        yield TraceHandle(trace_id)
        return
    from langfuse import propagate_attributes  # noqa: PLC0415
    from langfuse.langchain import CallbackHandler  # noqa: PLC0415

    # Langfuse metadata attributes are short strings.
    attrs = {k: str(v)[:200] for k, v in (metadata or {}).items() if v is not None}
    # propagate_attributes first: only observations created inside it carry the session / user /
    # tags, and that has to include the root observation.
    with (
        propagate_attributes(
            trace_name=name,
            session_id=session_id,
            user_id=user_id,
            tags=list(tags) or None,
            metadata=attrs or None,
        ),
        _client.start_as_current_observation(
            name=name, as_type="span", trace_context={"trace_id": trace_id}, input=input
        ) as root,
    ):
        handler = CallbackHandler()
        token = _callbacks_ctx.set((handler,))
        try:
            yield TraceHandle(trace_id, [handler], root)
        finally:
            _callbacks_ctx.reset(token)


class _NoopSpan:
    def update(self, **_: Any) -> None:
        return None


@contextmanager
def span(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,  # noqa: A002
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """A child observation of the current trace (RAG stage, OCR page, MCP tool call). A failing
    block marks the span as an error and re-raises; tracing itself never raises."""
    if _client is None:
        yield _NoopSpan()
        return
    with _client.start_as_current_observation(
        name=name,
        as_type=as_type,  # type: ignore[arg-type]
        input=input,
        metadata=metadata,
    ) as observation:
        yield observation


def score_trace(trace_id: str, name: str, value: float | str, comment: str | None = None) -> None:
    """Attach an eval score to a trace (no-op when tracing is off)."""
    if _client is None:
        return
    try:
        _client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        logger.warning("score_failed", extra={"score": name}, exc_info=True)
