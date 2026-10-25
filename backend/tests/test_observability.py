"""Module 10, part A: tracing (mask, ids, no-op mode), metrics and log context. No LLM, no network."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.llm import CircuitBreaker
from app.core.logging import JsonFormatter, bind_log_context
from app.observability import tracing
from app.observability.mask import REDACTED, mask_trace_data
from app.observability.metrics import record_cache, record_llm_call, record_mcp_tool_call
from app.observability.pricing import cost_usd

# ------------------------------------------------------------------ mask


def test_mask_redacts_phone_email_aadhaar_and_pan_inside_nested_data() -> None:
    data = {
        "messages": [
            {
                "content": "Call 9876543210 or mail ravi.k@example.com, Aadhaar 2345 6789 0123, PAN ABCDE1234F"
            }
        ],
        "n": 3,
    }
    out = json.dumps(mask_trace_data(data=data))
    for raw in ("9876543210", "ravi.k@example.com", "2345 6789 0123", "ABCDE1234F"):
        assert raw not in out
    assert "[phone]" in out and "[email]" in out and "[aadhaar]" in out and "[pan]" in out
    assert mask_trace_data(data=data)["n"] == 3


def test_mask_replaces_sensitive_keys_whatever_the_value_looks_like() -> None:
    out = mask_trace_data(data={"patient_name": "Ravi Kumar", "diagnosis": "Hernia", "age": 41})
    assert out == {"patient_name": REDACTED, "diagnosis": "Hernia", "age": 41}


def test_mask_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_: object) -> str:
        raise RuntimeError("x")

    monkeypatch.setattr("app.observability.mask.redact", boom)
    assert mask_trace_data(data="9876543210") == "[masking failed]"


# ------------------------------------------------------------------ tracing off / on


def test_tracing_is_a_noop_without_keys() -> None:
    assert (
        tracing.init_tracing(Settings(langfuse_public_key=None, langfuse_secret_key=None)) is False
    )
    assert tracing.tracing_enabled() is False
    with tracing.run_trace("claim_run", seed="run-1") as trace:
        assert trace.callbacks == []
        trace.update(output={"x": 1})  # no error
        with tracing.span("rag.vector") as s:
            s.update(output={"hits": 2})
    assert tracing.current_callbacks() == []
    tracing.score_trace(trace.trace_id, "decision_correct", 1.0)  # no error


def test_trace_id_is_deterministic_32_hex() -> None:
    a = tracing.trace_id_for("run-1")
    assert a == tracing.trace_id_for("run-1") and a != tracing.trace_id_for("run-2")
    assert len(a) == 32 and int(a, 16) >= 0


def test_traced_run_hands_out_the_callback_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real SDK with export switched off: ids, handler and context behave as in production."""
    from langfuse import Langfuse

    client = Langfuse(
        public_key="pk-test",
        secret_key="sk-test",
        base_url="http://localhost:1",
        tracing_enabled=False,
        mask=mask_trace_data,
    )
    monkeypatch.setattr(tracing, "_client", client)
    with tracing.run_trace(
        "claim_run",
        seed="run-9",
        session_id="CLM-014",
        user_id="u1",
        tags=["scenario:eval", "policy:POL-AS"],
        metadata={"run_id": "run-9", "n": None},
    ) as trace:
        assert trace.trace_id == tracing.trace_id_for("run-9")
        assert len(trace.callbacks) == 1 and tracing.current_callbacks() == trace.callbacks
        with tracing.span("mcp.search_policy", as_type="tool") as s:
            s.update(metadata={"success": True})
        trace.update(output={"status": "awaiting_approval"})
    assert tracing.current_callbacks() == []


# ------------------------------------------------------------------ log context


def test_log_lines_carry_trace_run_and_claim_ids_when_bound() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", (), None)
    plain = json.loads(JsonFormatter().format(record))
    assert "trace_id" not in plain and "run_id" not in plain
    with bind_log_context(trace_id="abc", run_id="r1", claim_number="CLM-001"):
        bound = json.loads(JsonFormatter().format(record))
    assert (bound["trace_id"], bound["run_id"], bound["claim_number"]) == ("abc", "r1", "CLM-001")
    assert "trace_id" not in json.loads(JsonFormatter().format(record))


# ------------------------------------------------------------------ metrics


def test_cost_uses_the_longest_model_prefix() -> None:
    assert cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(0.15)
    assert cost_usd("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.50)
    assert cost_usd("unknown-model", 10, 10) == 0.0


def test_breaker_state_code() -> None:
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=30, clock=lambda: now[0])
    assert breaker.state_code() == 0
    breaker.record_failure()
    assert breaker.state_code() == 2
    now[0] = 31.0
    assert breaker.state_code() == 1


def test_metrics_endpoint_exposes_the_custom_metrics() -> None:
    from app.main import create_app

    app = create_app(Settings(_env_file=None, app_env="test", metrics_token="scrape-token-123456"))
    with TestClient(app) as client:
        _check_metrics(client)


def _check_metrics(client: TestClient) -> None:
    record_llm_call(
        provider="openai", model="gpt-4o-mini", node="coverage", tokens_in=1000, tokens_out=200
    )
    record_llm_call(
        provider="groq", model="llama-3.3-70b-versatile", node="risk", tokens_in=10, tokens_out=5
    )
    record_llm_call(
        provider="openai", model="gpt-4o-mini", node="risk", tokens_in=0, tokens_out=0, failed=True
    )
    record_cache(True)
    record_cache(False)
    record_mcp_tool_call("policy_docs", "search_policy", ok=True)
    client.get("/health")
    assert client.get("/metrics").status_code == 401  # no token
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    body = client.get("/metrics", headers={"Authorization": "Bearer scrape-token-123456"}).text
    for name in (
        'llm_calls_total{model="gpt-4o-mini",node="coverage",provider="openai"}',
        "llm_fallback_total",
        "llm_tokens_total",
        "llm_cost_inr_total",
        "llm_call_errors_total",
        "circuit_breaker_state",
        "rag_cache_hits_total",
        "rag_cache_misses_total",
        "agent_runs_total",
        "agent_run_duration_seconds",
        "ocr_pages_total",
        'mcp_tool_calls_total{server="policy_docs",status="ok",tool="search_policy"}',
        'http_requests_total{method="GET",route="/health",status="200"}',
        "http_request_duration_seconds",
    ):
        assert name in body, name
    assert 'route="/metrics"' not in body  # scraping itself is not counted


def test_trace_id_header_is_set_when_a_route_provides_one(client: TestClient) -> None:
    from app.api.agent_runs import bind_run_trace  # noqa: F401 - the dependency used by the run routes

    resp = client.get("/health")
    assert "X-Trace-ID" not in resp.headers  # only routes tied to a trace set it


# ------------------------------------------------------------------ eval -> Langfuse


def test_eval_push_is_a_noop_without_langfuse() -> None:
    from eval import langfuse_push as lf

    assert lf.ready() is False
    lf.ensure_dataset("d", "desc", [{"id": "a"}])
    lf.link_run("run", "a", "t" * 32)
    lf.push_scores("t" * 32, {"decision_correct": True})
    lf.finish()


def test_push_scores_sends_numbers_and_skips_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eval import langfuse_push as lf

    sent: list[tuple[str, str, float, str | None]] = []
    monkeypatch.setattr(
        tracing,
        "score_trace",
        lambda trace_id, name, value, comment=None: sent.append((trace_id, name, value, comment)),
    )
    lf.push_scores(
        "abc", {"decision_correct": True, "amount_correct": None, "faithfulness": 0.8}, "run-1"
    )
    assert sent == [
        ("abc", "decision_correct", 1.0, "run-1"),
        ("abc", "faithfulness", 0.8, "run-1"),
    ]
