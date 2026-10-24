"""Document type detection: keyword rules first, an LLM only when the rules are unsure."""

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.intake.schemas import DocumentText
from app.models.enums import DocType

logger = logging.getLogger(__name__)

# Strong phrases per type. Each distinct phrase found adds one point.
KEYWORDS: dict[DocType, tuple[str, ...]] = {
    DocType.DISCHARGE_SUMMARY: (
        "discharge summary",
        "condition at discharge",
        "course in hospital",
        "discharge advice",
        "history of presenting illness",
        "summary of hospital stay",
        "advice on discharge",
        "procedure performed",
        "length of stay",
        "follow-up",
    ),
    DocType.HOSPITAL_BILL: (
        "final inpatient bill",
        "final bill",
        "inpatient bill",
        "invoice",
        "tax invoice",
        "sub total",
        "subtotal",
        "gst @",
        "amount (rs)",
        "rate (rs)",
        "bill no",
        "bill date",
        "room rent",
        "net payable",
    ),
    DocType.CLAIM_FORM: (
        "claim form",
        "claimant",
        "name of insured",
        "relationship to insured",
        "policy number",
        "policy_number",
        "pre_existing_conditions",
        "is_accident_case",
        "claimed_amount",
        "declaration",
    ),
}
# One decisive phrase is worth more than a stray word that every hospital document contains.
TITLE_RE: dict[DocType, re.Pattern[str]] = {
    DocType.DISCHARGE_SUMMARY: re.compile(
        r"^\s*(?:final\s+)?discharge\s+(?:summary|card)\b", re.I | re.M
    ),
    DocType.HOSPITAL_BILL: re.compile(
        r"^\s*(?:final\s+)?(?:inpatient\s+)?(?:bill|invoice)\b", re.I | re.M
    ),
    DocType.CLAIM_FORM: re.compile(r"^\s*(?:health\s+insurance\s+)?claim\s+form\b", re.I | re.M),
}
TITLE_BONUS = 3
MIN_SCORE = 3  # below this the rules are not sure
MIN_MARGIN = 2  # the winner must lead the runner-up by this much
EXCERPT_CHARS = 3000

LLMClassifier = Callable[[str], Awaitable[DocType | None]]


@dataclass(frozen=True)
class Classification:
    doc_type: DocType
    method: str  # "rules" | "llm" | "unknown"
    scores: dict[str, int]


def score_text(text: str) -> dict[DocType, int]:
    lowered = text.lower()
    scores = {
        doc_type: sum(1 for phrase in phrases if phrase in lowered)
        for doc_type, phrases in KEYWORDS.items()
    }
    for doc_type, pattern in TITLE_RE.items():
        if pattern.search(text):
            scores[doc_type] += TITLE_BONUS
    return scores


def classify_by_rules(text: str) -> tuple[DocType | None, dict[DocType, int]]:
    """The winning type if the evidence is clear, else None."""
    scores = score_text(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score >= MIN_SCORE and best_score - second_score >= MIN_MARGIN:
        return best, scores
    return None, scores


async def classify_document(
    document: DocumentText, llm_classifier: LLMClassifier | None = None
) -> Classification:
    """Rules first. `llm_classifier` (given a text excerpt) is called only when they are unsure;
    the caller is responsible for masking PII in what that callable sends to a model."""
    text = "\n".join(p.text for p in document.pages[:2])
    doc_type, scores = classify_by_rules(text)
    shown = {t.value: s for t, s in scores.items()}
    if doc_type is not None:
        return Classification(doc_type, "rules", shown)
    if llm_classifier is not None:
        try:
            guessed = await llm_classifier(text[:EXCERPT_CHARS])
        except Exception:  # noqa: BLE001 - classification must never crash intake
            logger.warning("classify_llm_failed", exc_info=True)
            guessed = None
        if guessed is not None:
            return Classification(guessed, "llm", shown)
    return Classification(DocType.OTHER, "unknown", shown)
