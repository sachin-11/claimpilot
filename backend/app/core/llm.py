"""LLM factory: OpenAI primary, Groq fallback, circuit breaker, per-call provider logging.

Use `get_chat_model()` / `get_embeddings()`; do not instantiate providers anywhere else.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from uuid import UUID

import groq
import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    BaseCallbackHandler,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult, LLMResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field

from app.core.config import Settings, get_settings, has_real_key
from app.observability.metrics import record_llm_call

logger = logging.getLogger(__name__)

# Only these trigger the fallback: rate limit (429), timeouts, connection errors, 5xx.
# (APITimeoutError subclasses APIConnectionError; InternalServerError is any 5xx.)
# 400 / 401 / 404 / validation errors are deliberately NOT here: they are our bug or our
# config, and switching provider would only hide them.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
)


class CircuitOpenError(Exception):
    """Raised by the primary model while the breaker is open; triggers the fallback."""


class CircuitBreaker:
    """Tiny in-process circuit breaker for the primary provider.

    closed:    calls go to primary. `failure_threshold` failures within `window_seconds` open it.
    open:      primary is skipped (calls raise CircuitOpenError, so they go straight to the
               fallback) for `cooldown_seconds`.
    half-open: after the cooldown one probe goes to primary. Success closes the breaker;
               a failure re-opens it immediately.

    State is per process (not shared across workers), which is fine: each worker learns on its
    own within a few calls. The clock is injectable so tests don't sleep.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._window = window_seconds
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._failures: deque[float] = deque()
        self._opened_at: float | None = None
        self._half_open = False
        self._lock = threading.Lock()  # sync LangChain calls can run in worker threads

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if self._clock() - self._opened_at >= self._cooldown:
                self._opened_at = None
                self._half_open = True
                self._failures.clear()
                logger.info("llm_circuit_half_open")
                return False
            return True

    def state_code(self) -> int:
        """0 closed, 1 half-open, 2 open (for the metrics gauge; does not change the state)."""
        with self._lock:
            if self._opened_at is None:
                return 1 if self._half_open else 0
            return 1 if self._clock() - self._opened_at >= self._cooldown else 2

    def record_failure(self) -> None:
        with self._lock:
            now = self._clock()
            if self._half_open:
                self._trip(now)
                return
            while self._failures and now - self._failures[0] > self._window:
                self._failures.popleft()
            self._failures.append(now)
            if len(self._failures) >= self._threshold:
                self._trip(now)

    def record_success(self) -> None:
        with self._lock:
            if self._opened_at is not None:  # a call that started before the trip; ignore
                return
            self._failures.clear()
            if self._half_open:
                self._half_open = False
                logger.info("llm_circuit_closed")

    def _trip(self, now: float) -> None:
        self._opened_at = now
        self._half_open = False
        self._failures.clear()
        logger.warning("llm_circuit_open", extra={"cooldown_seconds": self._cooldown})


class BreakerChatOpenAI(ChatOpenAI):
    """ChatOpenAI that reports to / obeys a CircuitBreaker.

    The breaker lives on the primary model, so `.with_fallbacks()` stays a plain LangChain
    fallback chain: an open breaker just makes the primary raise CircuitOpenError, which the
    fallback chain handles like any other retryable error. `max_retries` is the OpenAI SDK's own
    backoff, so the breaker counts calls that failed *after* those retries were used up.
    """

    breaker: Any = Field(default=None, exclude=True)

    def _guard(self) -> None:
        if self.breaker is not None and self.breaker.is_open:
            raise CircuitOpenError("primary LLM circuit is open")

    def _record(self, ok: bool) -> None:
        if self.breaker is not None:
            self.breaker.record_success() if ok else self.breaker.record_failure()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._guard()
        try:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except RETRYABLE_ERRORS:
            self._record(False)
            raise
        self._record(True)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._guard()
        try:
            result = await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except RETRYABLE_ERRORS:
            self._record(False)
            raise
        self._record(True)
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._guard()
        try:
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        except RETRYABLE_ERRORS:
            self._record(False)
            raise
        self._record(True)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._guard()
        try:
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except RETRYABLE_ERRORS:
            self._record(False)
            raise
        self._record(True)


class ProviderLoggingHandler(BaseCallbackHandler):
    """Logs which provider/model actually served each call, with latency and token usage."""

    def __init__(self) -> None:
        self._started: dict[UUID, tuple[float, str, str]] = {}
        self._nodes: dict[UUID, str] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        invocation_params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # LangChain puts ls_provider / ls_model_name in `metadata` for every chat model;
        # invocation_params is the fallback (ChatGroq doesn't put the model name there).
        params, meta = invocation_params or {}, metadata or {}
        provider = str(meta.get("ls_provider") or params.get("_type", "unknown"))
        provider = provider.removesuffix("-chat")
        model = str(meta.get("ls_model_name") or params.get("model") or "unknown")
        self._started[run_id] = (time.perf_counter(), provider, model)
        # the LangGraph node ("coverage:<id>|model:<id>" inside a subgraph -> "coverage")
        namespace = str(meta.get("langgraph_checkpoint_ns") or meta.get("langgraph_node") or "")
        self._nodes[run_id] = namespace.split("|")[0].split(":")[0] or "none"

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        started = self._started.pop(run_id, None)
        if started is None:
            return
        t0, provider, model = started
        node = self._nodes.pop(run_id, "none")
        usage: dict[str, Any] = {}
        try:
            message = response.generations[0][0].message  # type: ignore[union-attr]
            usage = dict(getattr(message, "usage_metadata", None) or {})
        except (IndexError, AttributeError):
            pass
        if not usage:
            usage = dict((response.llm_output or {}).get("token_usage") or {})
        tokens_in = int(usage.get("input_tokens", usage.get("prompt_tokens")) or 0)
        tokens_out = int(usage.get("output_tokens", usage.get("completion_tokens")) or 0)
        record_llm_call(
            provider=provider, model=model, node=node, tokens_in=tokens_in, tokens_out=tokens_out
        )
        logger.info(
            "llm_call",
            extra={
                "llm_provider": provider,
                "llm_model": model,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
                "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
                "total_tokens": usage.get("total_tokens"),
            },
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        started = self._started.pop(run_id, None)
        if started is None:
            return
        t0, provider, model = started
        node = self._nodes.pop(run_id, "none")
        record_llm_call(
            provider=provider, model=model, node=node, tokens_in=0, tokens_out=0, failed=True
        )
        logger.warning(
            "llm_call_failed",
            extra={
                "llm_provider": provider,
                "llm_model": model,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "error_type": type(error).__name__,
            },
        )


def build_chat_model(
    settings: Settings,
    breaker: CircuitBreaker,
    temperature: float = 0,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Build the primary (+ fallback) chat model. `get_chat_model` is the app-facing wrapper.

    With a fallback the result is a RunnableWithFallbacks. It still supports `bind_tools` and
    `with_structured_output`, which LangChain applies to both models.
    """
    handler = ProviderLoggingHandler()
    callbacks = [handler, *kwargs.pop("callbacks", [])]
    fallback_enabled = has_real_key(settings.groq_api_key)

    primary = BreakerChatOpenAI(
        model=settings.primary_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
        callbacks=callbacks,
        # With no fallback an open breaker would only turn slow errors into instant ones.
        breaker=breaker if fallback_enabled else None,
        **kwargs,
    )
    if not fallback_enabled:
        logger.warning("llm_fallback_disabled", extra={"reason": "GROQ_API_KEY not set"})
        return primary

    fallback = ChatGroq(
        model=settings.fallback_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
        callbacks=callbacks,
        **kwargs,
    )
    return primary.with_fallbacks(
        [fallback], exceptions_to_handle=(*RETRYABLE_ERRORS, CircuitOpenError)
    )


def _without_thinking(messages: Any) -> Any:
    """Message list minus the thinking blocks of earlier assistant turns (a thinking-off call has no
    use for them, and their signatures no longer match once the request differs)."""
    if not isinstance(messages, list):
        return messages
    cleaned: list[Any] = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"}
            for b in content
        ):
            kept = [
                b
                for b in content
                if not (isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"})
            ]
            message = message.model_copy(update={"content": kept or ""})
        cleaned.append(message)
    return cleaned


class EvalChatAnthropic(ChatAnthropic):
    """ChatAnthropic whose structured output is a reliable forced tool call.

    With thinking on the tool call can't be forced, so a reply without it fails to parse. The
    structured-output calls (extraction, final answers) need no reasoning, so they run with thinking
    off; the tool loops keep adaptive thinking. (Claude's native JSON-schema mode was tried first
    and rejects the extraction schema: too many union-typed fields.)
    """

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        method: str = "function_calling",
        **kwargs: Any,
    ) -> Any:
        plain = self.model_copy(update={"thinking": {"type": "disabled"}, "betas": None})
        structured = super(EvalChatAnthropic, plain).with_structured_output(
            schema, include_raw=include_raw, method="function_calling", **kwargs
        )
        return RunnableLambda(_without_thinking) | structured


def build_anthropic_chat_model(
    settings: Settings, **kwargs: Any
) -> Runnable[LanguageModelInput, BaseMessage]:
    """Claude for the evaluation runs (no breaker, no fallback: an eval must fail loudly).

    No `temperature` on purpose: Claude 5.x models return a 400 for any non-default value, so the
    model's own default is used. `get_chat_model` stays OpenAI (+ Groq) for the app itself.
    """
    if not has_real_key(settings.anthropic_api_key):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (needed for the evaluation runs)")
    kwargs.pop("temperature", None)
    callbacks = [ProviderLoggingHandler(), *kwargs.pop("callbacks", [])]
    return EvalChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout_seconds,
        max_tokens=kwargs.pop("max_tokens", 4096),
        callbacks=callbacks,
        # LangChain rebuilds the assistant turns of a tool loop (block order), which breaks the
        # signature of Claude 5.x thinking blocks ("bound to a different conversation", a 400).
        # drop_block lets the request through without the mismatched blocks instead of failing.
        betas=["thinking-binding-controls-2026-08-01"],
        thinking={"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
        **kwargs,
    )


def get_anthropic_chat_model(**kwargs: Any) -> Runnable[LanguageModelInput, BaseMessage]:
    return build_anthropic_chat_model(get_settings(), **kwargs)


_breaker: CircuitBreaker | None = None


def get_breaker(settings: Settings | None = None) -> CircuitBreaker:
    """Process-wide breaker, shared by every model `get_chat_model` hands out."""
    global _breaker
    if _breaker is None:
        s = settings or get_settings()
        _breaker = CircuitBreaker(
            failure_threshold=s.breaker_failure_threshold,
            window_seconds=s.breaker_window_seconds,
            cooldown_seconds=s.breaker_cooldown_seconds,
        )
    return _breaker


def get_chat_model(
    temperature: float = 0, **kwargs: Any
) -> Runnable[LanguageModelInput, BaseMessage]:
    settings = get_settings()
    return build_chat_model(settings, get_breaker(settings), temperature, **kwargs)


def get_embeddings() -> OpenAIEmbeddings:
    # No cross-provider fallback for embeddings on purpose: every model has its own vector space
    # (and dimension), so vectors from a fallback model can't be compared with the ones already
    # stored in pgvector. Mixing them would silently wreck retrieval. Fail loudly instead.
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        timeout=settings.outbound_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
