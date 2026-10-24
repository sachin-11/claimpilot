"""Golden-set tooling: normalisation, candidate generation guards, human review bookkeeping."""

import json
import random
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from eval.rag.build_candidates import (
    CandidateBuilder,
    CandidateDraft,
    ChunkInfo,
    ParaphraseDraft,
    UnanswerableDrafts,
    build_candidates,
    is_eligible,
    pick_pairs,
    scaled_counts,
    write_candidates,
)
from eval.rag.common import QUESTIONS_PATH, normalize_text, read_jsonl
from eval.rag.review import (
    accept_candidate,
    next_question_id,
    pending_candidates,
    progress,
    record_skip,
)
from eval.rag.schemas import GoldenQuestion


def chunk(n: int, section: str = "6 Waiting Period", content: str | None = None, tokens: int = 200):
    return ChunkInfo(
        id=f"00000000-0000-0000-0000-{n:012d}",
        section_path=section,
        content=content
        or f"Clause {n}: cataract is covered after 24 months of continuous coverage. "
        f"Co-payment is 5% of the admissible amount and room rent is up to Rs.5,000/- per day. {n}",
        page_start=n,
        page_end=n,
        token_count=tokens,
        chunk_index=n,
    )


class SchemaChat:
    """Fake chat model: answers each structured-output schema with a canned callable."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self):
        return self

    def with_structured_output(self, schema, include_raw=True, method=None):
        parent = self

        class Runner:
            async def ainvoke(self, messages):
                parent.calls.append(schema.__name__)
                return {
                    "raw": AIMessage(content=""),
                    "parsed": parent.responses[schema](messages),
                    "parsing_error": None,
                }

        return Runner()


def golden(**kw) -> GoldenQuestion:
    base = {
        "id": "cand-numeric-001",
        "policy_codes": ["POL-AS"],
        "question": "What is the waiting period for cataract?",
        "category": "numeric",
        "reference_answer": "24 months.",
        "must_contain": ["24 months"],
    }
    base.update(kw)
    return GoldenQuestion(**base)


# ---------------------------------------------------------------- normalisation / schema


def test_normalize_text_matches_the_ways_numbers_are_written() -> None:
    assert normalize_text("24 Months") == normalize_text("24 months")
    assert normalize_text("5 %") == normalize_text("5%") == "5%"
    assert normalize_text("Rs. 5,000/-") == normalize_text("Rs.5000") == normalize_text("Rs 5000")
    assert normalize_text("₹ 10,000") == normalize_text("Rs.10000")
    assert normalize_text("  two   spaces\n") == "two spaces"
    assert normalize_text("24 months") != normalize_text("36 months")  # digits are never altered


def test_golden_question_schema_is_strict() -> None:
    assert golden().verified_by_human is False  # candidates are unverified by default
    with pytest.raises(ValidationError):
        golden(category="trivia")
    with pytest.raises(ValidationError):
        golden(unknown_field=1)
    with pytest.raises(ValidationError):
        golden(question="")


def test_only_real_qa_material_is_eligible() -> None:
    assert is_eligible(chunk(1))
    assert not is_eligible(chunk(2, section="Annexure - C : List of Day Care Procedures"))
    assert not is_eligible(chunk(3, section="11 Get in touch with us:"))
    assert not is_eligible(chunk(4, tokens=10))
    assert not is_eligible(chunk(5, content="| a | b |\n" * 20))


# ---------------------------------------------------------------- candidate generation


async def test_draft_keeps_only_must_contain_strings_that_really_are_in_the_clause() -> None:
    draft = CandidateDraft(
        question="How long is the cataract waiting period?",
        reference_answer="24 months.",
        must_contain=["24 Months", "36 months", "co-payment is 5 %", "x" * 90],
    )
    chat = SchemaChat({CandidateDraft: lambda m: draft})
    builder = CandidateBuilder(chat, random.Random(1))

    candidate = await builder.from_chunks("numeric", [chunk(1)], "POL-AS", avoid=[])

    assert candidate is not None
    assert candidate.must_contain == ["24 Months", "co-payment is 5 %"]  # 36 months: not in source
    assert candidate.reference_chunk_ids == [chunk(1).id]
    assert candidate.reference_sections == ["6 Waiting Period"]
    assert candidate.verified_by_human is False and candidate.id == "cand-numeric-001"


async def test_draft_without_any_verifiable_must_contain_is_dropped() -> None:
    draft = CandidateDraft(question="q?", reference_answer="a", must_contain=["99 years"])
    builder = CandidateBuilder(SchemaChat({CandidateDraft: lambda m: draft}), random.Random(1))
    assert await builder.from_chunks("numeric", [chunk(1)], "POL-AS", avoid=[]) is None


async def test_a_failing_llm_call_does_not_stop_the_batch() -> None:
    def boom(messages):
        raise RuntimeError("rate limited")

    builder = CandidateBuilder(SchemaChat({CandidateDraft: boom}), random.Random(1))
    assert await builder.from_chunks("coverage", [chunk(1)], "POL-AS", avoid=[]) is None


async def test_full_candidate_build_mix_and_flags() -> None:
    counter = {"n": 0}

    def draft(messages):
        counter["n"] += 1
        return CandidateDraft(
            question=f"Question number {counter['n']}?",
            reference_answer="24 months.",
            must_contain=["24 months"],
        )

    chat = SchemaChat(
        {
            CandidateDraft: draft,
            ParaphraseDraft: lambda m: ParaphraseDraft(question=f"yaar, kab? #{len(chat.calls)}"),
            UnanswerableDrafts: lambda m: UnanswerableDrafts(
                questions=["What is my premium at age 45?", "Which hospitals in Pune are network?"]
            ),
        }
    )
    chunks = [
        chunk(i, section=f"{i} Waiting Period" if i < 5 else "7 EXCLUSIONS") for i in range(1, 12)
    ]
    chunks.append(chunk(50, section="Annexure - C : List"))
    counts = {
        "numeric": 4,
        "coverage": 3,
        "exclusion": 2,
        "multi_clause": 2,
        "unanswerable": 2,
        "paraphrase": 2,
    }

    candidates = await build_candidates(chunks, "POL-AS", chat, counts, seed=7)

    by_category = {
        c: [x for x in candidates if x.category == c] for c in {x.category for x in candidates}
    }
    assert len(by_category["numeric"]) == 4 and len(by_category["exclusion"]) == 2
    assert len(by_category["multi_clause"]) == 2
    assert all(len(c.reference_chunk_ids) == 2 for c in by_category["multi_clause"])
    assert len(by_category["paraphrase"]) == 2
    assert all(c.expect_insufficient_context for c in by_category["unanswerable"])
    assert len(by_category["injection"]) == 4
    assert all(c.forbidden_phrases for c in by_category["injection"])
    assert all(c.verified_by_human is False for c in candidates)
    assert len({c.id for c in candidates}) == len(candidates)
    assert "Annexure" not in " ".join(s for c in candidates for s in c.reference_sections)


def test_pick_pairs_are_related_and_distinct() -> None:
    chunks = [chunk(i, section="6 Waiting Period") for i in range(1, 6)] + [
        chunk(20, section="4. COVERAGE"),
        chunk(21, section="4. COVERAGE"),
    ]
    pairs = pick_pairs(chunks, 5, random.Random(3))
    assert pairs
    for first, second in pairs:
        assert first.id != second.id
        assert first.section_path.split(" ")[0] == second.section_path.split(" ")[0]
    assert len({tuple(sorted((a.id, b.id))) for a, b in pairs}) == len(pairs)


def test_candidates_can_never_be_written_to_questions_jsonl(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="questions.jsonl"):
        write_candidates(QUESTIONS_PATH, [golden()], force=True)
    with pytest.raises(ValueError, match="questions.jsonl"):
        write_candidates(tmp_path / "questions.jsonl", [golden()], force=True)

    target = tmp_path / "candidates.jsonl"
    write_candidates(target, [golden()])
    assert (
        json.loads(target.read_text(encoding="utf-8").splitlines()[0])["verified_by_human"] is False
    )
    with pytest.raises(FileExistsError):
        write_candidates(target, [golden()])
    write_candidates(target, [golden(), golden(id="cand-numeric-002")], force=True)
    assert len(read_jsonl(target)) == 2


def test_scaled_counts() -> None:
    assert scaled_counts(1.0)["numeric"] == 19
    assert scaled_counts(0.1)["injection" if "injection" in scaled_counts(0.1) else "numeric"] >= 1


# ---------------------------------------------------------------- review bookkeeping


def paths(tmp_path: Path):
    return tmp_path / "candidates.jsonl", tmp_path / "questions.jsonl", tmp_path / "skipped.json"


def test_accept_appends_verified_records_with_sequential_ids(tmp_path: Path) -> None:
    _, questions, _ = paths(tmp_path)

    first = accept_candidate(golden(), questions)
    second = accept_candidate(
        golden(id="cand-numeric-002", question="Another question?", category="coverage"),
        questions,
        edits={"reference_answer": "edited answer", "must_contain": ["edited"]},
    )

    records = read_jsonl(questions)
    assert [r["id"] for r in records] == ["q-001", "q-002"] == [first.id, second.id]
    assert all(r["verified_by_human"] is True for r in records)
    assert records[1]["reference_answer"] == "edited answer" and records[1]["must_contain"] == [
        "edited"
    ]
    assert next_question_id(records) == "q-003"
    assert progress(questions)["numeric"] == (1, 12) and progress(questions)["coverage"] == (1, 8)


def test_accepting_the_same_question_twice_is_refused(tmp_path: Path) -> None:
    _, questions, _ = paths(tmp_path)
    accept_candidate(golden(), questions)
    with pytest.raises(ValueError, match="already"):
        accept_candidate(
            golden(id="cand-numeric-009", question="what is the WAITING period for cataract?"),
            questions,
        )
    assert len(read_jsonl(questions)) == 1


def test_pending_excludes_accepted_and_skipped(tmp_path: Path) -> None:
    candidates, questions, skipped = paths(tmp_path)
    write_candidates(
        candidates,
        [
            golden(id="c1", question="First question?"),
            golden(id="c2", question="Second question?"),
            golden(id="c3", question="Third question?"),
        ],
    )

    accept_candidate(golden(id="c1", question="First question?"), questions)
    record_skip("c2", skipped)

    pending = pending_candidates(candidates, questions, skipped)
    assert [c.id for c in pending] == ["c3"]
    again = pending_candidates(candidates, questions, skipped, include_skipped=True)
    assert [c.id for c in again] == ["c2", "c3"]


# ---------------------------------------------------------------- extend_candidates

CLAUSES_CHUNK = (
    "7 EXCLUSIONS\n\n"
    "7.4 Change-of-Gender treatments: (Code- Excl07)\n\n"
    "Expenses related to any treatment, including surgical management, to change "
    "characteristics of the body to those of the opposite sex.\n\n"
    "7.5 Cosmetic or plastic Surgery: (Code- Excl08)\n\n"
    "Expenses for cosmetic or plastic surgery or any treatment to change appearance unless for "
    "reconstruction following an Accident, Burn(s) or Cancer.\n\n"
    "7.6 Hazardous or Adventure sports: (Code- Excl09)\n\n"
    "Expenses related to any treatment necessitated due to participation as a professional in "
    "hazardous or adventure sports, including rafting."
)


def test_chunks_are_split_into_clause_units() -> None:
    from eval.rag.extend_candidates import split_clause_units

    units = split_clause_units(chunk(1, section="7 EXCLUSIONS", content=CLAUSES_CHUNK))

    assert [u.clause for u in units] == ["7.4", "7.5", "7.6"]
    assert all(u.id == chunk(1).id and u.section_path == "7 EXCLUSIONS" for u in units)
    assert "cosmetic" in units[1].content and "Cosmetic" not in units[2].content
    short = chunk(2, content="Too short to ask about.")
    assert split_clause_units(short) == []


def test_a_clause_is_covered_only_by_a_candidate_whose_must_contain_is_in_that_clause() -> None:
    from eval.rag.extend_candidates import coverage_report, split_clause_units

    base = chunk(1, section="7 EXCLUSIONS", content=CLAUSES_CHUNK)
    units = split_clause_units(base)
    cosmetic = golden(
        id="c1",
        question="Is cosmetic surgery excluded?",
        category="exclusion",
        reference_chunk_ids=[base.id],
        must_contain=["cosmetic or plastic surgery"],
    )
    wrong_chunk = golden(
        id="c2", question="Other?", reference_chunk_ids=[chunk(9).id], must_contain=["rafting"]
    )
    injection = golden(
        id="c3",
        question="Ignore all?",
        category="injection",
        reference_chunk_ids=[base.id],
        must_contain=["rafting"],
    )

    covered, uncovered = coverage_report(units, [cosmetic, wrong_chunk, injection])

    assert covered == 1 and uncovered == ["7.4", "7.6"]


def test_related_pairs_link_clauses_that_share_rare_words() -> None:
    from eval.rag.extend_candidates import related_pairs

    def unit(n: int, text: str):
        return ChunkInfo(
            id=f"id-{n}",
            section_path=f"{n} Section",
            content=text,
            page_start=n,
            page_end=n,
            token_count=30,
            chunk_index=n,
            clause=f"{n}.1",
        )

    units = [
        unit(
            1, "cataract treatment is subject to a limit per eye in one policy year " + "alpha " * 3
        ),
        unit(2, "specific waiting period applies to cataract and hernia conditions " + "beta " * 3),
        unit(3, "ambulance charges road ambulance maximum per hospitalisation " + "gamma " * 3),
        unit(4, "free look period fifteen days premium refund proposal " + "delta " * 3),
    ]

    pairs = related_pairs(units, 3)

    assert (units[0], units[1]) in pairs  # share "cataract"
    assert all(a.id != b.id for a, b in pairs)


async def test_extension_adds_unverified_candidates_without_touching_existing_ones() -> None:
    from eval.rag.extend_candidates import (
        AnswerCheck,
        NearMissDrafts,
        extend_candidates,
    )

    counter = {"n": 0}

    def draft(messages):
        counter["n"] += 1
        return CandidateDraft(
            question=f"New question number {counter['n']}?",
            reference_answer="answer",
            must_contain=["cosmetic or plastic surgery"],
        )

    chat = SchemaChat(
        {
            CandidateDraft: draft,
            ParaphraseDraft: lambda m: ParaphraseDraft(
                question=f"yaar ye kya hai #{len(chat.calls)}"
            ),
            NearMissDrafts: lambda m: NearMissDrafts(
                questions=["Is dental implant covered?", "What is the premium for a 50 year old?"]
            ),
            # the second near-miss turns out to be answered by the policy: it must be filtered out
            AnswerCheck: lambda m: AnswerCheck(answered_in_policy="dental" not in m[-1].content),
        }
    )
    chunks = [chunk(1, section="7 EXCLUSIONS", content=CLAUSES_CHUNK)]
    existing = [
        golden(
            id="cand-numeric-007",
            question="Existing question?",
            reference_chunk_ids=[chunks[0].id],
            must_contain=["Hazardous"],
        ),
    ]

    new = await extend_candidates(chunks, existing, "POL-AS", chat, max_units=10, seed=1)

    assert new and all(c.verified_by_human is False for c in new)
    assert "Is dental implant covered?" in [c.question for c in new]
    assert "What is the premium for a 50 year old?" not in [c.question for c in new]
    near_miss = [c for c in new if c.category == "unanswerable"]
    assert near_miss and all(c.expect_insufficient_context for c in near_miss)
    assert {c.category for c in new} >= {"exclusion", "paraphrase", "injection", "unanswerable"}
    assert len([c for c in new if c.category == "injection"]) == 4
    assert all(c.forbidden_phrases for c in new if c.category == "injection")
    ids = [c.id for c in [*existing, *new]]
    assert len(set(ids)) == len(ids)  # numbering continues, no collisions
    assert "cand-numeric-007" in ids and not any(c.id == "cand-numeric-007" for c in new)
    # clause 7.6 already had a candidate (its must_contain 'Hazardous' is in that clause)
    asked_about = " ".join(c.reference_answer for c in new)
    assert "answer" in asked_about


def test_definition_only_candidates_are_reviewed_last(tmp_path: Path) -> None:
    from eval.rag.review import from_definitions_only

    candidates, questions, skipped = paths(tmp_path)
    write_candidates(
        candidates,
        [
            golden(id="c1", question="What does X mean?", reference_sections=["3. DEFINITIONS"]),
            golden(id="c2", question="Waiting period?", reference_sections=["6 Waiting Period"]),
            golden(
                id="c3",
                question="Mixed?",
                reference_sections=["3. DEFINITIONS", "6 Waiting Period"],
            ),
            golden(id="c4", question="No section?", reference_sections=[]),
        ],
    )

    pending = pending_candidates(candidates, questions, skipped)

    assert [c.id for c in pending] == ["c2", "c3", "c4", "c1"]
    assert from_definitions_only(pending[-1]) and not from_definitions_only(pending[1])
