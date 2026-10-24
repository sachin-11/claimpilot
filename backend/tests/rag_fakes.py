"""Fakes for the RAG tests: no real embedding or chat API is ever called."""

import hashlib
import math
import re
from collections.abc import Callable
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, AIMessageChunk

from app.models.policy import EMBEDDING_DIM
from app.rag.config import RagSettings
from app.rag.schemas import Citation, PolicyAnswer, RetrievedChunk

CLAUSE_ID_RE = re.compile(r'<clause id="([0-9a-f-]{36})"')


def vec(**axes: float) -> list[float]:
    """Unit vector from named axes: vec(a=1) -> e0, vec(a=3, b=4) -> (0.6, 0.8, ...)."""
    names = {c: i for i, c in enumerate("abcdefghij")}
    v = [0.0] * EMBEDDING_DIM
    for name, weight in axes.items():
        v[names[name]] = weight
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def hashed_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 for b in digest] * (EMBEDDING_DIM // 32)


class FakeEmbeddings:
    def __init__(self, mapping: dict[str, list[float]] | None = None, *, fail: bool = False):
        self.mapping = mapping or {}
        self.fail = fail
        self.calls: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("embedding service down")
        return self.mapping.get(text) or hashed_vector(text)


def rag_settings(**overrides: Any) -> RagSettings:
    return RagSettings(
        _env_file=None, cache_prefix=overrides.pop("cache_prefix", "test:rag"), **overrides
    )


class FakeChat:
    """Stands in for get_chat_model(): scripted structured answers, plain replies, token streams.

    `structured` is a PolicyAnswer or a callable(clause_ids) -> PolicyAnswer, so a test can cite
    whatever chunks were really retrieved.
    """

    def __init__(
        self,
        *,
        structured: PolicyAnswer | Callable[[list[UUID]], PolicyAnswer] | None = None,
        text: str = "standalone question",
        stream_tokens: list[str] | None = None,
        model_name: str = "fake-model",
    ):
        self.structured = structured
        self.text = text
        self.stream_tokens = stream_tokens or []
        self.model_name = model_name
        self.calls: list[tuple[str, list]] = []

    def __call__(self) -> "FakeChat":  # usable directly as a chat_model_factory
        return self

    def with_structured_output(
        self, schema: Any, include_raw: bool = True, method: str | None = None
    ):
        return _Structured(self)

    async def ainvoke(self, messages: list) -> AIMessage:
        self.calls.append(("text", messages))
        return AIMessage(content=self.text)

    async def astream(self, messages: list):
        self.calls.append(("stream", messages))
        for token in self.stream_tokens:
            yield AIMessageChunk(content=token, response_metadata={"model_name": self.model_name})

    def calls_of(self, kind: str) -> list[list]:
        return [m for k, m in self.calls if k == kind]


class _Structured:
    def __init__(self, parent: FakeChat):
        self.parent = parent

    async def ainvoke(self, messages: list) -> dict:
        self.parent.calls.append(("structured", messages))
        ids = [UUID(i) for i in CLAUSE_ID_RE.findall(messages[-1].content)]
        answer = self.parent.structured
        if callable(answer):
            answer = answer(ids)
        return {
            "raw": AIMessage(content="", response_metadata={"model_name": self.parent.model_name}),
            "parsed": answer,
            "parsing_error": None,
        }


def cites_first(
    text: str = "Answer.", quote: str = "exact words"
) -> Callable[[list[UUID]], PolicyAnswer]:
    def build(ids: list[UUID]) -> PolicyAnswer:
        citations = [Citation(chunk_id=ids[0], supporting_text=quote)] if ids else []
        return PolicyAnswer(answer=text, citations=citations, confidence="high")

    return build


def make_chunk(n: int, **kwargs: Any) -> RetrievedChunk:
    defaults = {
        "chunk_id": UUID(int=n),
        "policy_code": "POL-A",
        "section_path": f"Section {n}",
        "page_start": n,
        "page_end": n,
        "content": f"clause text {n}",
    }
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)
