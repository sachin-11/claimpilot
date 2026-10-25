"""MOCK_LLM: the deterministic fake model and embeddings used for UI end-to-end tests and demos."""

import json
import math
from pathlib import Path
from uuid import uuid4

import pymupdf
import pytest
from agent_helpers import agent_deps, initial_state, make_agent_world, mcp_world
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from mcp_helpers import FakeRag
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_graph
from app.core.config import Settings, get_settings, production_problems
from app.core.mock_llm import (
    EMBEDDING_DIMENSIONS,
    MockChat,
    MockEmbeddings,
    embed_text,
    mock_extraction,
    waiting_period,
)
from app.intake.schemas import LLMExtraction
from app.rag.answerer import INSUFFICIENT_TEXT, Answerer, build_messages
from app.rag.config import RagSettings
from app.rag.schemas import RetrievedChunk

DATA = get_settings().data_dir
TRUTH = {
    r["claim_id"]: r
    for r in json.loads((DATA / "golden" / "extraction_truth.json").read_text("utf-8"))
}


# ------------------------------------------------------------------ never in production


def test_mock_llm_cannot_be_enabled_in_production() -> None:
    with pytest.raises(ValidationError, match="MOCK_LLM"):
        Settings(_env_file=None, app_env="production", mock_llm=True)
    assert Settings(_env_file=None, app_env="development", mock_llm=True).mock_llm is True
    problems = production_problems(Settings(_env_file=None, app_env="development", mock_llm=True))
    assert "MOCK_LLM is enabled" in problems  # also named by the production check


def test_the_factories_return_the_fakes_only_when_switched_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core import llm

    monkeypatch.setattr(llm, "get_settings", lambda: Settings(_env_file=None, mock_llm=True))
    assert isinstance(llm.get_chat_model(temperature=0, callbacks=[]), MockChat)
    assert isinstance(llm.get_embeddings(), MockEmbeddings)
    monkeypatch.setattr(
        llm, "get_settings", lambda: Settings(_env_file=None, mock_llm=False, openai_api_key="sk-x")
    )
    assert not isinstance(llm.get_chat_model(), MockChat)
    assert not isinstance(llm.get_embeddings(), MockEmbeddings)


def test_the_guard_refuses_production() -> None:
    from app.core.mock_llm import assert_allowed

    assert_allowed(False)
    with pytest.raises(RuntimeError, match="never be used in production"):
        assert_allowed(True)


# ------------------------------------------------------------------ embeddings


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_embeddings_are_deterministic_normalised_and_lexical() -> None:
    emb = MockEmbeddings()
    one = await emb.aembed_query("initial waiting period of thirty days")
    assert len(one) == EMBEDDING_DIMENSIONS == len(embed_text("x"))
    assert one == await emb.aembed_query("initial waiting period of thirty days")  # repeatable
    assert math.isclose(sum(v * v for v in one), 1.0, rel_tol=1e-9)
    near = await emb.aembed_query("what is the initial waiting period")
    far = await emb.aembed_query("dental implant exclusion cosmetic surgery")
    assert cosine(one, near) > cosine(one, far) + 0.3
    docs = await emb.aembed_documents(["room rent limit", "dental implant"])
    assert len(docs) == 2 and docs[0] != docs[1]


# ------------------------------------------------------------------ extraction


def documents_block(claim_id: str) -> str:
    blocks = []
    for doc_type, name in (
        ("hospital_bill", "hospital_bill.pdf"),
        ("discharge_summary", "discharge_summary.pdf"),
    ):
        path = DATA / "claims" / claim_id / name
        if not path.exists():
            continue
        with pymupdf.open(path) as pdf:
            for number, page in enumerate(pdf, start=1):
                blocks.append(
                    f'<document type="{doc_type}" page="{number}">\n{page.get_text()}\n</document>'
                )
    return "\n\n".join(blocks)


COMPLETE = sorted(
    claim_id
    for claim_id, row in TRUTH.items()
    if (DATA / "claims" / claim_id / "discharge_summary.pdf").exists()
    and (DATA / "claims" / claim_id / "hospital_bill.pdf").exists()
    and not row["expected_missing_documents"]
    and row["fields"].get("bill_total") is not None
)


def test_there_are_golden_claims_to_check_the_extraction_against() -> None:
    assert len(COMPLETE) >= 30


@pytest.mark.parametrize("claim_id", COMPLETE)
def test_extraction_reads_the_generated_documents_like_the_truth(claim_id: str) -> None:
    truth = TRUTH[claim_id]
    fields = truth["fields"]
    out = mock_extraction(
        [SystemMessage("x"), HumanMessage(f"Documents:\n\n{documents_block(claim_id)}")]
    )
    assert isinstance(out, LLMExtraction)
    if "date_mismatch" not in (truth.get("expected_consistency_issues") or []):
        assert out.admission_date == fields["admission_date"]  # (else the documents disagree)
        assert out.discharge_date == fields["discharge_date"]
    assert out.diagnosis_text == fields["diagnosis_text"]
    assert (out.icd10_code or "").upper() == (fields.get("icd10_code") or "").upper()
    assert out.hospital_name == fields["hospital_name"]
    assert out.bill_total == float(fields["bill_total"])
    assert out.bill_subtotal == float(fields["bill_subtotal"])
    assert (out.taxes or 0) == float(fields["taxes"] or 0)
    assert [i.amount for i in out.line_items] == [float(i["amount"]) for i in truth["line_items"]]
    assert [i.description for i in out.line_items] == [
        i["description"] for i in truth["line_items"]
    ]
    assert {o.doc_type for o in out.observations} == {"hospital_bill", "discharge_summary"}
    assert {s.field for s in out.field_sources} >= {
        "admission_date",
        "bill_total",
        "diagnosis_text",
    }
    assert all(s.page and s.confidence == "high" for s in out.field_sources)


def test_extraction_maps_line_items_to_categories_and_finds_the_room() -> None:
    out = mock_extraction([HumanMessage(f"Documents:\n\n{documents_block('CLM-014')}")])
    by_text = {i.description: i.category for i in out.line_items}
    assert by_text["Room Rent - Deluxe Room"] == "room"
    assert by_text["Surgeon Fee - Laparoscopic Appendectomy"] == "surgeon"
    assert by_text["Operation Theatre Charges"] == "ot"
    assert by_text["Tablet Ondansetron 4 mg"] == "medicines"
    assert by_text["IV Cannula, Syringes & Infusion Sets"] == "consumables"
    assert by_text["USG Abdomen"] == "diagnostics"
    assert out.room_type == "Deluxe Room" and out.room_rent_per_day == 8400.0
    assert out.patient_age == 27 and out.gender == "Female"


def test_extraction_without_documents_returns_nulls_not_errors() -> None:
    out = mock_extraction([HumanMessage("Documents:\n\n")])
    assert out.bill_total is None and out.line_items == [] and out.observations == []


# ------------------------------------------------------------------ waiting periods in clause text


@pytest.mark.parametrize(
    ("check", "text", "key", "expected"),
    [
        (
            "initial_waiting",
            "Any illness is not covered during the first thirty days.",
            None,
            (30, "days"),
        ),
        ("initial_waiting", "first 45 days from the start", None, (45, "days")),
        (
            "specific_disease_waiting",
            "hernia: 24 months; cataract: 12 months.",
            "cataract",
            (12, "months"),
        ),
        (
            "pre_existing",
            "covered only after 36 months of continuous coverage",
            None,
            (36, "months"),
        ),
        ("initial_waiting", "nothing about periods here", None, None),
    ],
)
def test_waiting_period_is_read_from_the_clause(check, text, key, expected) -> None:
    assert waiting_period(check, text, key) == expected


# ------------------------------------------------------------------ the whole agent flow


async def test_the_agent_graph_runs_end_to_end_on_the_mock(session: AsyncSession) -> None:
    world = await make_agent_world(session)
    async with mcp_world(session, FakeRag(world.chunk)) as provider:
        deps = agent_deps(session, provider, MockChat())
        graph = build_graph(deps, InMemorySaver())
        config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 40}
        await graph.ainvoke(initial_state(world), config)
        state = (await graph.aget_state(config)).values
    assert state["errors"] == []
    assert state["completed_steps"][:4] == ["intake", "coverage", "risk", "decision"] or {
        "intake", "coverage", "risk", "decision"
    } <= set(state["completed_steps"])  # fmt: skip
    cited = [c for check in state["coverage"] for c in check["cited_clauses"]]
    assert cited and {c["chunk_id"] for c in cited} == {str(world.chunk.id)}
    decision = state["decision"]
    assert decision["decision"] in {"approve", "partial_approve", "flag_for_review"}
    assert decision["reasons"] and state["models_used"]["coverage"] == "mock-llm"
    assert state["risk"]["summary"]


# ------------------------------------------------------------------ policy answers


def chunk(n: int, section: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        policy_code="POL-AS",
        section_path=section,
        page_start=n + 5,
        page_end=n + 5,
        content=content,
        chunk_index=n,
    )


CLAUSES = [
    chunk(
        1,
        "6 Waiting Period > 6.2",
        "Any illness is not covered during the first thirty days from the policy start date.",
    ),
    chunk(
        2, "7 Exclusions > 7.3", "Expenses for obesity or weight-control treatment are excluded."
    ),
    chunk(3, "4 Coverage", "Room rent is limited to one percent of the sum insured per day."),
]


async def test_the_answer_cites_the_clause_that_matches_the_question() -> None:
    answerer = Answerer(lambda: MockChat(), RagSettings())
    outcome = await answerer.answer("What is the initial waiting period of thirty days?", CLAUSES)
    answer = outcome.answer
    assert not answer.insufficient_context and "thirty days" in answer.answer
    assert [c.chunk_id for c in answer.citations] == [CLAUSES[0].chunk_id]
    assert answer.citations[0].section_path == "6 Waiting Period > 6.2"
    assert outcome.model_used == "mock-llm"

    obesity = await answerer.answer("Is obesity treatment covered?", CLAUSES)
    assert obesity.answer.citations[0].chunk_id == CLAUSES[1].chunk_id


async def test_an_unrelated_question_is_not_covered() -> None:
    answerer = Answerer(lambda: MockChat(), RagSettings())
    outcome = await answerer.answer("Does the policy cover space travel insurance?", CLAUSES)
    assert outcome.answer.insufficient_context and outcome.answer.answer == INSUFFICIENT_TEXT
    assert outcome.answer.citations == []


async def test_the_streamed_answer_carries_clause_markers_or_the_sentinel() -> None:
    chat = MockChat()
    covered = "".join(
        [
            piece.content
            async for piece in chat.astream(
                build_messages("initial waiting period thirty days", CLAUSES, streaming=True)
            )
        ]
    )
    assert covered.strip().endswith("[1]") and "thirty days" in covered
    missing = "".join(
        [
            piece.content
            async for piece in chat.astream(
                build_messages("space travel insurance", CLAUSES, streaming=True)
            )
        ]
    )
    assert missing.startswith("INSUFFICIENT:")


async def test_classification_rewriting_and_unknown_schemas() -> None:
    chat = MockChat()
    classified = await chat.ainvoke(
        [
            SystemMessage("Classify the document below. Answer with exactly one word"),
            HumanMessage("DISCHARGE SUMMARY ..."),
        ]
    )
    assert classified.content == "discharge_summary"
    rewritten = await chat.ainvoke(
        [
            SystemMessage("rewrite"),
            HumanMessage("Conversation:\n...\n\nLatest question: And cataract?"),
        ]
    )
    assert rewritten.content == "And cataract?"
    with pytest.raises(NotImplementedError):
        chat.with_structured_output(dict)
    assert isinstance(ToolMessage(content="[]", tool_call_id="1"), ToolMessage)


_ = Path
