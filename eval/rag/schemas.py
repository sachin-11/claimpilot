from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "numeric", "coverage", "exclusion", "multi_clause", "unanswerable", "paraphrase", "injection"
]

# Target mix of the verified golden set (~40 questions).
TARGET_MIX: dict[str, int] = {
    "numeric": 12,
    "coverage": 8,
    "exclusion": 6,
    "multi_clause": 5,
    "unanswerable": 4,
    "paraphrase": 3,
    "injection": 2,
}
MIN_VERIFIED_QUESTIONS = 20


class GoldenQuestion(BaseModel):
    """One record of questions.jsonl or candidates.jsonl (candidates: verified_by_human=false)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    policy_codes: list[str]
    question: str = Field(min_length=3)
    category: Category
    reference_answer: str
    reference_sections: list[str] = Field(default_factory=list)
    reference_chunk_ids: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    expect_insufficient_context: bool = False
    verified_by_human: bool = False
    # Injection questions only: phrases whose presence means the model complied with the attack.
    forbidden_phrases: list[str] = Field(default_factory=list)
