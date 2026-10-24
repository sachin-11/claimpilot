from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    policy_code: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    content: str
    vector_rank: int | None = None  # 1-based; None = not in the vector top-k
    keyword_rank: int | None = None  # 1-based; None = not in the keyword top-k
    fused_score: float = 0.0
    rerank_score: float | None = None


class Citation(BaseModel):
    chunk_id: UUID
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    supporting_text: str = Field(
        default="", description="Exact words from the clause, <= 200 chars"
    )


class PolicyAnswer(BaseModel):
    """What the LLM must return (structured output)."""

    answer: str = Field(description="Answer in plain language, grounded only in the clauses")
    citations: list[Citation] = Field(default_factory=list)
    insufficient_context: bool = Field(
        default=False, description="True if the clauses do not cover the question"
    )
    confidence: Literal["low", "medium", "high"] = "medium"


class LLMCitation(BaseModel):
    """Citation as the LLM writes it. The model points at a clause by its short number n
    (1, 2, 3 ... as shown in the prompt), which is far easier to copy than a UUID. Everything is
    lenient on purpose: a bad value drops the citation, it must never fail the request."""

    clause_n: int | None = Field(
        default=None, description="The n attribute of the clause you relied on (1, 2, 3 ...)"
    )
    chunk_id: str | None = Field(default=None, description="Optional: the clause id attribute")
    supporting_text: str = Field(
        default="", description="Exact words from the clause, <= 200 chars"
    )


class LLMPolicyAnswer(BaseModel):
    """The structured-output schema sent to the LLM; converted to PolicyAnswer afterwards."""

    answer: str = Field(description="Answer in plain language, grounded only in the clauses")
    citations: list[LLMCitation] = Field(default_factory=list)
    insufficient_context: bool = Field(
        default=False, description="True if the clauses do not cover the question"
    )
    confidence: Literal["low", "medium", "high"] = "medium"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    policy_codes: list[str] | None = Field(default=None, max_length=10)
    top_n: int | None = Field(default=None, ge=1, le=20)


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    policy_codes: list[str] | None = Field(default=None, max_length=10)
    chat_history: list[ChatTurn] | None = Field(default=None, max_length=20)
    use_cache: bool = True


class PolicyAnswerResult(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool
    confidence: Literal["low", "medium", "high"]
    chunks_used: list[RetrievedChunk]
    model_used: str | None = None
    cache_hit: bool = False
    rewritten_query: str | None = None  # set only when a follow-up was rewritten
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    """One server-sent event: citations | token | done | error."""

    event: Literal["citations", "token", "done", "error"]
    data: dict | list | str
