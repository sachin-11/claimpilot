import json
from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from rag_fakes import make_chunk

from app.core.config import Settings
from app.main import create_app
from app.rag.answerer import AnswerGenerationError
from app.rag.schemas import Citation, PolicyAnswerResult, StreamEvent


class StubService:
    """Records calls; returns canned results. The real service is covered in test_rag_service."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail = False

    async def search(self, query, policy_codes=None, top_n=None):
        self.calls.append(("search", query, policy_codes, top_n))
        return [make_chunk(1, rerank_score=0.8, vector_rank=2, keyword_rank=1, fused_score=0.03)]

    async def answer(self, query, policy_codes=None, chat_history=None, use_cache=True):
        self.calls.append(("answer", query, policy_codes, chat_history, use_cache))
        if self.fail:
            raise AnswerGenerationError("LLM call failed: boom")
        chunk = make_chunk(1)
        return PolicyAnswerResult(
            answer="24 months.",
            citations=[
                Citation(
                    chunk_id=chunk.chunk_id, section_path="Section 1", page_start=1, page_end=1
                )
            ],
            insufficient_context=False,
            confidence="high",
            chunks_used=[chunk],
            model_used="fake-model",
            timings_ms={"generate": 1.0},
        )

    async def answer_stream(
        self, query, policy_codes=None, chat_history=None, use_cache=True
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(("stream", query, policy_codes, chat_history, use_cache))
        yield StreamEvent(
            event="citations", data=[{"chunk_id": str(UUID(int=1)), "section_path": "S"}]
        )
        yield StreamEvent(event="token", data="24 ")
        yield StreamEvent(event="token", data="months.")
        yield StreamEvent(event="done", data={"answer": "24 months.", "cache_hit": False})


@pytest.fixture
def api() -> tuple[TestClient, StubService]:
    app = create_app(Settings(_env_file=None, app_env="test"))
    stub = StubService()
    with TestClient(app) as client:
        app.state.rag_service = stub
        yield client, stub


def parse_sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_search_endpoint_exposes_ranks_and_scores(api) -> None:
    client, stub = api

    response = client.post(
        "/policies/search", json={"query": "cataract", "policy_codes": ["POL-AS"], "top_n": 3}
    )

    assert response.status_code == 200
    chunk = response.json()[0]
    assert {"vector_rank", "keyword_rank", "fused_score", "rerank_score", "content"} <= set(chunk)
    assert (chunk["vector_rank"], chunk["keyword_rank"], chunk["rerank_score"]) == (2, 1, 0.8)
    assert stub.calls == [("search", "cataract", ["POL-AS"], 3)]


def test_query_endpoint_returns_answer_with_citations_and_timings(api) -> None:
    client, stub = api
    body = {
        "query": "and for cataract?",
        "policy_codes": ["POL-AS"],
        "chat_history": [{"role": "user", "content": "hi"}],
        "use_cache": False,
    }

    response = client.post("/policies/query", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "24 months." and data["citations"][0]["section_path"] == "Section 1"
    assert data["model_used"] == "fake-model" and data["cache_hit"] is False
    assert data["timings_ms"] == {"generate": 1.0}
    call = stub.calls[0]
    assert call[1] == "and for cataract?" and call[4] is False
    assert call[3][0].role == "user"


def test_query_endpoint_maps_generation_failure_to_502(api) -> None:
    client, stub = api
    stub.fail = True
    response = client.post("/policies/query", json={"query": "cataract"})
    assert response.status_code == 502
    assert "boom" in response.json()["error"]["message"]


def test_request_validation(api) -> None:
    client, _ = api
    assert client.post("/policies/search", json={"query": ""}).status_code == 422
    assert client.post("/policies/query", json={"query": "x" * 2001}).status_code == 422
    assert (
        client.post(
            "/policies/query",
            json={"query": "ok?", "chat_history": [{"role": "system", "content": "x"}]},
        ).status_code
        == 422
    )
    assert client.post("/policies/search", json={"query": "ok?", "top_n": 99}).status_code == 422


def test_stream_endpoint_emits_citations_tokens_done(api) -> None:
    client, stub = api

    with client.stream("POST", "/policies/query/stream", json={"query": "cataract?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        body = "".join(response.iter_text())

    events = parse_sse(body)
    assert [name for name, _ in events] == ["citations", "token", "token", "done"]
    assert events[0][1][0]["section_path"] == "S"
    assert "".join(data for name, data in events if name == "token") == "24 months."
    assert events[-1][1]["answer"] == "24 months."
    assert stub.calls[0][0] == "stream"


def test_service_is_reported_unavailable_when_it_cannot_be_built(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("no OPENAI_API_KEY")

    monkeypatch.setattr("app.api.rag.build_rag_service", broken)
    app = create_app(Settings(_env_file=None, app_env="test"))
    with TestClient(app) as client:
        response = client.post("/policies/search", json={"query": "cataract"})
    assert response.status_code == 503
