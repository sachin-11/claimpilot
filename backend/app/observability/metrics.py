"""Prometheus metrics (`GET /metrics`). One module owns every metric so the names and labels stay
in one place; the rest of the code calls the small `record_*` helpers.

Per process: the API exposes its own registry, and a separate worker process (Part C) exposes
its own. Label values are bounded (provider, model, node, route template, tool): never ids."""

import time
from collections.abc import Callable
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.observability.pricing import cost_usd
from app.observability.usage import record_run, record_spend

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
LLM_CALLS = Counter("llm_calls_total", "LLM calls", ["provider", "model", "node"])
LLM_ERRORS = Counter("llm_call_errors_total", "Failed LLM calls", ["provider", "model", "node"])
LLM_FALLBACK = Counter(
    "llm_fallback_total", "LLM calls answered by the fallback provider", ["provider", "model"]
)
LLM_TOKENS = Counter(
    "llm_tokens_total", "LLM tokens", ["provider", "model", "direction"]
)  # direction: input | output
LLM_COST_INR = Counter("llm_cost_inr_total", "Estimated LLM cost in INR", ["provider", "model"])
BREAKER_STATE = Gauge("circuit_breaker_state", "LLM circuit breaker: 0 closed, 1 half-open, 2 open")
RAG_CACHE_HITS = Counter("rag_cache_hits_total", "RAG semantic cache hits")
RAG_CACHE_MISSES = Counter("rag_cache_misses_total", "RAG semantic cache misses")
AGENT_RUNS = Counter("agent_runs_total", "Agent runs by outcome", ["status"])
AGENT_RUN_DURATION = Histogram(
    "agent_run_duration_seconds",
    "Agent run duration until it pauses or finishes",
    buckets=(5, 10, 20, 30, 45, 60, 90, 120, 300, 600),
)
JOBS = Counter("worker_jobs_total", "Worker jobs by outcome", ["job", "status"])
OCR_PAGES = Counter("ocr_pages_total", "Pages read, by method", ["method"])
MCP_TOOL_CALLS = Counter(
    "mcp_tool_calls_total", "MCP tool calls made by the agents", ["server", "tool", "status"]
)

PROVIDERS_FALLBACK = frozenset({"groq"})  # provider names that mean "the fallback served this"


def usd_inr_rate() -> float:
    from app.core.config import get_settings  # noqa: PLC0415 - avoid an import cycle at load

    return get_settings().usd_inr_rate


def llm_cost_inr(model: str, tokens_in: int, tokens_out: int) -> float:
    return cost_usd(model, tokens_in, tokens_out) * usd_inr_rate()


def record_llm_call(
    *,
    provider: str,
    model: str,
    node: str,
    tokens_in: int,
    tokens_out: int,
    failed: bool = False,
) -> float:
    """Count one LLM call; returns its estimated cost in INR (0 for a failed call)."""
    LLM_CALLS.labels(provider, model, node).inc()
    if failed:
        LLM_ERRORS.labels(provider, model, node).inc()
        return 0.0
    if provider in PROVIDERS_FALLBACK:
        LLM_FALLBACK.labels(provider, model).inc()
    LLM_TOKENS.labels(provider, model, "input").inc(tokens_in)
    LLM_TOKENS.labels(provider, model, "output").inc(tokens_out)
    inr = llm_cost_inr(model, tokens_in, tokens_out)
    LLM_COST_INR.labels(provider, model).inc(inr)
    record_spend(
        provider=provider, model=model, inr=inr, tokens_in=tokens_in, tokens_out=tokens_out
    )
    return inr


def record_agent_run(status: str, seconds: float) -> None:
    AGENT_RUNS.labels(status).inc()
    record_run(status)
    AGENT_RUN_DURATION.observe(seconds)


def record_mcp_tool_call(server: str, tool: str, ok: bool) -> None:
    MCP_TOOL_CALLS.labels(server, tool, "ok" if ok else "error").inc()


def record_job(job: str, status: str) -> None:
    JOBS.labels(job, status).inc()


def record_cache(hit: bool) -> None:
    (RAG_CACHE_HITS if hit else RAG_CACHE_MISSES).inc()


def record_ocr_page(method: str) -> None:
    OCR_PAGES.labels(method).inc()


def bind_breaker(state: Callable[[], int]) -> None:
    BREAKER_STATE.set_function(lambda: float(state()))


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


class RequestTimer:
    """Context for the HTTP middleware: `observe(route, status)` once the response is known."""

    def __init__(self, method: str) -> None:
        self.method = method
        self.started = time.perf_counter()

    def observe(self, route: str, status: int) -> None:
        HTTP_REQUESTS.labels(self.method, route, str(status)).inc()
        HTTP_DURATION.labels(self.method, route).observe(time.perf_counter() - self.started)


def route_template(scope: dict[str, Any]) -> str:
    """The matched route path ("/claims/{claim_id}/process"), never the concrete URL."""
    route = scope.get("route")
    return getattr(route, "path", None) or "unmatched"
