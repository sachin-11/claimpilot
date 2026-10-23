"""LLM fallback + circuit breaker tests. Provider calls are patched; nothing hits the network."""

from typing import Any

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import Settings
from app.core.llm import CircuitBreaker, build_chat_model

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    return cls("boom", response=httpx.Response(status, request=_REQUEST), body=None)


def _result(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Calls:
    """Patches ChatOpenAI / ChatGroq `_generate` and counts how often each was hit."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.primary = 0
        self.fallback = 0
        self.primary_error: Exception | None = None
        self.fallback_error: Exception | None = None
        calls = self

        def primary(*args: Any, **kwargs: Any):
            calls.primary += 1
            if calls.primary_error:
                raise calls.primary_error
            return _result("from-primary")

        def fallback(*args: Any, **kwargs: Any):
            calls.fallback += 1
            if calls.fallback_error:
                raise calls.fallback_error
            return _result("from-fallback")

        monkeypatch.setattr(ChatOpenAI, "_generate", primary)
        monkeypatch.setattr(ChatGroq, "_generate", fallback)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=3, window_seconds=60, cooldown_seconds=30, clock=clock)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    return Calls(monkeypatch)


def test_primary_is_used_when_healthy(settings: Settings, breaker: CircuitBreaker, calls: Calls):
    model = build_chat_model(settings, breaker)
    assert model.invoke("hi").content == "from-primary"
    assert (calls.primary, calls.fallback) == (1, 0)


def test_rate_limit_falls_back(settings: Settings, breaker: CircuitBreaker, calls: Calls):
    calls.primary_error = _status_error(openai.RateLimitError, 429)
    model = build_chat_model(settings, breaker)
    assert model.invoke("hi").content == "from-fallback"
    assert (calls.primary, calls.fallback) == (1, 1)


@pytest.mark.parametrize(
    "error",
    [
        _status_error(openai.InternalServerError, 503),
        openai.APIConnectionError(request=_REQUEST),
        openai.APITimeoutError(request=_REQUEST),
    ],
    ids=["5xx", "connection", "timeout"],
)
def test_other_retryable_errors_fall_back(
    error: Exception, settings: Settings, breaker: CircuitBreaker, calls: Calls
):
    calls.primary_error = error
    assert build_chat_model(settings, breaker).invoke("hi").content == "from-fallback"


@pytest.mark.parametrize(
    "error",
    [
        _status_error(openai.BadRequestError, 400),
        _status_error(openai.AuthenticationError, 401),
        ValueError("bad input"),
    ],
    ids=["400", "401", "validation"],
)
def test_non_retryable_errors_surface_without_fallback(
    error: Exception, settings: Settings, breaker: CircuitBreaker, calls: Calls
):
    calls.primary_error = error
    model = build_chat_model(settings, breaker)
    with pytest.raises(type(error)):
        model.invoke("hi")
    assert calls.fallback == 0
    # And they don't count towards opening the breaker.
    assert not breaker.is_open
    breaker.record_failure()
    breaker.record_failure()
    assert not breaker.is_open


async def test_async_path_falls_back(
    settings: Settings, breaker: CircuitBreaker, monkeypatch: pytest.MonkeyPatch
):
    async def primary(*args: Any, **kwargs: Any):
        raise _status_error(openai.RateLimitError, 429)

    async def fallback(*args: Any, **kwargs: Any):
        return _result("from-fallback")

    monkeypatch.setattr(ChatOpenAI, "_agenerate", primary)
    monkeypatch.setattr(ChatGroq, "_agenerate", fallback)
    result = await build_chat_model(settings, breaker).ainvoke("hi")
    assert result.content == "from-fallback"


def test_breaker_opens_after_n_failures_and_skips_primary(
    settings: Settings, breaker: CircuitBreaker, calls: Calls, clock: FakeClock
):
    calls.primary_error = _status_error(openai.RateLimitError, 429)
    model = build_chat_model(settings, breaker)

    for _ in range(3):  # threshold = 3
        assert model.invoke("hi").content == "from-fallback"
    assert calls.primary == 3
    assert breaker.is_open

    # Open: straight to fallback, primary is not even attempted.
    assert model.invoke("hi").content == "from-fallback"
    assert calls.primary == 3

    # After the cooldown, primary is probed again. It recovered, so the breaker closes.
    clock.now += 31
    calls.primary_error = None
    assert model.invoke("hi").content == "from-primary"
    assert calls.primary == 4
    assert not breaker.is_open


def test_failed_probe_reopens_immediately(
    settings: Settings, breaker: CircuitBreaker, calls: Calls, clock: FakeClock
):
    calls.primary_error = _status_error(openai.RateLimitError, 429)
    model = build_chat_model(settings, breaker)
    for _ in range(3):
        model.invoke("hi")
    assert breaker.is_open

    clock.now += 31  # cooldown over: half-open, one probe allowed
    assert model.invoke("hi").content == "from-fallback"  # probe fails -> fallback serves
    assert calls.primary == 4
    assert breaker.is_open  # one failed probe is enough to re-open

    model.invoke("hi")
    assert calls.primary == 4  # skipped again


def test_old_failures_age_out_of_the_window(breaker: CircuitBreaker, clock: FakeClock):
    breaker.record_failure()
    breaker.record_failure()
    clock.now += 61  # outside the 60s window
    breaker.record_failure()
    assert not breaker.is_open


def test_success_resets_the_failure_count(breaker: CircuitBreaker):
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert not breaker.is_open


def test_breaker_closes_after_cooldown(breaker: CircuitBreaker, clock: FakeClock):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.is_open
    clock.now += 29
    assert breaker.is_open
    clock.now += 2
    assert not breaker.is_open


def test_no_groq_key_returns_primary_only(
    settings: Settings, breaker: CircuitBreaker, calls: Calls, caplog: pytest.LogCaptureFixture
):
    settings = settings.model_copy(update={"groq_api_key": None})
    calls.primary_error = _status_error(openai.RateLimitError, 429)
    with caplog.at_level("WARNING"):
        model = build_chat_model(settings, breaker)
    assert any(r.message == "llm_fallback_disabled" for r in caplog.records)
    assert not hasattr(model, "fallbacks")
    with pytest.raises(openai.RateLimitError):
        model.invoke("hi")
    assert not breaker.is_open  # breaker is off without a fallback


def test_placeholder_groq_key_counts_as_missing(settings: Settings, breaker: CircuitBreaker):
    settings = settings.model_copy(update={"groq_api_key": SecretStr("gsk-replace-me")})
    assert isinstance(build_chat_model(settings, breaker), ChatOpenAI)


def test_logs_which_provider_served_the_call(
    settings: Settings, breaker: CircuitBreaker, calls: Calls, caplog: pytest.LogCaptureFixture
):
    calls.primary_error = _status_error(openai.RateLimitError, 429)
    with caplog.at_level("INFO", logger="app.core.llm"):
        build_chat_model(settings, breaker).invoke("hi")

    failed = next(r for r in caplog.records if r.message == "llm_call_failed")
    assert (failed.llm_provider, failed.llm_model) == ("openai", "gpt-4o-mini")
    assert failed.error_type == "RateLimitError"

    served = next(r for r in caplog.records if r.message == "llm_call")
    assert (served.llm_provider, served.llm_model) == ("groq", "llama-3.3-70b-versatile")
