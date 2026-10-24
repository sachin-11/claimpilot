"""Human review of golden-set candidates.

    make eval-review      # = docker compose exec backend python -m eval.rag.review (needs a TTY)

For each pending candidate you see the question, the reference answer, the must_contain strings
and the SOURCE CHUNK TEXT. Check the answer against the source, then:
    a = accept as is     e = edit then accept     s = skip (not shown again)
    d = show full source q = quit (progress is saved after every decision)
Accepted items are appended to golden/questions.jsonl with verified_by_human=true.
"""

import argparse
import asyncio
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.db import create_engine
from app.core.redis import create_redis
from app.models import PolicyChunk
from app.rag.config import RagSettings
from app.rag.service import build_rag_service
from eval.rag.common import (
    CANDIDATES_PATH,
    QUESTIONS_PATH,
    SKIPPED_PATH,
    append_jsonl,
    normalize_text,
    read_jsonl,
)
from eval.rag.schemas import TARGET_MIX, GoldenQuestion

WIDTH = 100
SEPARATOR = " | "


# ---------------------------------------------------------------- pure helpers (unit-tested)


def next_question_id(existing: list[dict]) -> str:
    numbers = [
        int(q["id"][2:]) for q in existing if q["id"].startswith("q-") and q["id"][2:].isdigit()
    ]
    return f"q-{max(numbers, default=0) + 1:03d}"


def load_skipped(path: Path = SKIPPED_PATH) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def record_skip(candidate_id: str, path: Path = SKIPPED_PATH) -> None:
    skipped = load_skipped(path) | {candidate_id}
    path.write_text(json.dumps(sorted(skipped), indent=2), encoding="utf-8")


def pending_candidates(
    candidates_path: Path = CANDIDATES_PATH,
    questions_path: Path = QUESTIONS_PATH,
    skipped_path: Path = SKIPPED_PATH,
    *,
    include_skipped: bool = False,
) -> list[GoldenQuestion]:
    """Candidates not yet accepted (same question text) and, by default, not skipped."""
    accepted = {normalize_text(q["question"]) for q in read_jsonl(questions_path)}
    skipped = set() if include_skipped else load_skipped(skipped_path)
    pending = [
        c
        for c in (GoldenQuestion(**r) for r in read_jsonl(candidates_path))
        if normalize_text(c.question) not in accepted and c.id not in skipped
    ]
    # Definition-only questions are usually glossary lookups: show them last (stable sort).
    return sorted(pending, key=from_definitions_only)


def from_definitions_only(candidate: GoldenQuestion) -> bool:
    """True when every reference section is in "3. DEFINITIONS" (a weak source for a question)."""
    sections = candidate.reference_sections
    return bool(sections) and all(s.lower().startswith("3. definitions") for s in sections)


def accept_candidate(
    candidate: GoldenQuestion,
    questions_path: Path = QUESTIONS_PATH,
    edits: dict | None = None,
) -> GoldenQuestion:
    """Append the (optionally edited) candidate to questions.jsonl as human-verified."""
    existing = read_jsonl(questions_path)
    record = candidate.model_copy(update={**(edits or {}), "verified_by_human": True})
    record = GoldenQuestion(**{**record.model_dump(), "id": next_question_id(existing)})
    if any(normalize_text(q["question"]) == normalize_text(record.question) for q in existing):
        raise ValueError("this question is already in questions.jsonl")
    append_jsonl(questions_path, record.model_dump())
    return record


def progress(questions_path: Path = QUESTIONS_PATH) -> dict[str, tuple[int, int]]:
    done = Counter(q["category"] for q in read_jsonl(questions_path))
    return {category: (done.get(category, 0), target) for category, target in TARGET_MIX.items()}


# ---------------------------------------------------------------- interactive part


def out(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(line, WIDTH, initial_indent=indent, subsequent_indent=indent) or indent
        for line in text.splitlines()
    )


async def fetch_chunks(
    session_factory: async_sessionmaker[AsyncSession], ids: list[str]
) -> dict[str, PolicyChunk]:
    if not ids:
        return {}
    async with session_factory() as session:
        rows = await session.execute(
            select(PolicyChunk).where(PolicyChunk.id.in_([UUID(i) for i in ids]))
        )
        return {str(c.id): c for c in rows.scalars()}


def show_candidate(
    candidate: GoldenQuestion,
    index: int,
    total: int,
    chunks: dict[str, PolicyChunk],
    retrieved: list[tuple[str, str]] | None,
    *,
    full: bool,
) -> None:
    out("\n" + "=" * WIDTH)
    out(f"[{index}/{total}] {candidate.id}   category: {candidate.category}")
    done = progress()
    out("verified so far: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in done.items()))
    out("-" * WIDTH)
    out(f"QUESTION  : {candidate.question}")
    out(f"REFERENCE : {candidate.reference_answer}")
    out(
        f"must_contain: {candidate.must_contain}   expect_insufficient_context: "
        f"{candidate.expect_insufficient_context}"
    )
    if candidate.forbidden_phrases:
        out(f"forbidden_phrases (compliance if present): {candidate.forbidden_phrases}")
    out(f"reference_sections: {candidate.reference_sections}")
    if from_definitions_only(candidate):
        out(
            "NOTE: source is only 3. DEFINITIONS. Prefer a question about the operative clause "
            "(coverage / waiting period / limit); skip if it is just a glossary lookup."
        )
    out("-" * WIDTH)

    if candidate.reference_chunk_ids:
        for chunk_id in candidate.reference_chunk_ids:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                out(f"SOURCE {chunk_id}: not found in the database (re-ingested?)")
                continue
            text = (
                chunk.content
                if full
                else chunk.content[:1200] + (" [...]" if len(chunk.content) > 1200 else "")
            )
            out(f"SOURCE [{chunk.section_path}] (p.{chunk.page_start}-{chunk.page_end}):")
            out(wrap(text))
    if retrieved is not None:
        out("What the retriever finds for this question (top 3) - check it is really NOT answered:")
        for section, snippet in retrieved:
            out(f"  - [{section}]")
            out(wrap(snippet[:400], "      "))
    elif candidate.category == "unanswerable":
        out("(retrieval aid unavailable)")


def ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    value = input(f"  {prompt}{shown}: ").strip()
    return value or default


def edit_candidate(candidate: GoldenQuestion, source_text: str) -> dict:
    out("  Edit (Enter keeps the value shown). Lists use ' | ' as separator.")
    edits: dict = {
        "question": ask("question", candidate.question),
        "reference_answer": ask("reference_answer", candidate.reference_answer),
    }
    must = ask("must_contain", SEPARATOR.join(candidate.must_contain))
    edits["must_contain"] = [m.strip() for m in must.split("|") if m.strip()]
    sections = ask("reference_sections", SEPARATOR.join(candidate.reference_sections))
    edits["reference_sections"] = [s.strip() for s in sections.split("|") if s.strip()]
    expect = ask(
        "expect_insufficient_context (y/n)", "y" if candidate.expect_insufficient_context else "n"
    )
    edits["expect_insufficient_context"] = expect.lower().startswith("y")
    if candidate.forbidden_phrases:
        forbidden = ask("forbidden_phrases", SEPARATOR.join(candidate.forbidden_phrases))
        edits["forbidden_phrases"] = [f.strip() for f in forbidden.split("|") if f.strip()]
    if source_text:
        for item in edits["must_contain"]:
            if normalize_text(item) not in normalize_text(source_text):
                out(f"  WARNING: must_contain item {item!r} is not in the source chunk text")
    return edits


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.rag.review", description=__doc__)
    parser.add_argument("--include-skipped", action="store_true")
    parser.add_argument("--category", help="only review this category")
    parser.add_argument("--no-retrieval", action="store_true", help="skip the retrieval aid")
    parser.add_argument("--database-url", help="use another database (e.g. an experiment DB)")
    args = parser.parse_args(argv)

    pending = pending_candidates(include_skipped=args.include_skipped)
    if args.category:
        pending = [c for c in pending if c.category == args.category]
    if not pending:
        out("Nothing to review: no pending candidates. Run `make eval-candidates` first?")
        return 0

    settings = get_settings()
    engine = create_engine(args.database_url or settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    redis = create_redis(settings.redis_url)
    service = None
    if not args.no_retrieval:
        try:
            service = build_rag_service(
                session_factory, None, RagSettings().model_copy(update={"reranker": "none"})
            )
        except Exception as exc:  # noqa: BLE001 - the aid is optional
            out(f"(retrieval aid disabled: {exc})")

    accepted = skipped = 0
    try:
        for number, candidate in enumerate(pending, start=1):
            chunks = await fetch_chunks(session_factory, candidate.reference_chunk_ids)
            source_text = "\n".join(c.content for c in chunks.values())
            retrieved = None
            if service is not None and candidate.category in ("unanswerable", "injection"):
                try:
                    found = await service.search(
                        candidate.question, candidate.policy_codes, top_n=3
                    )
                    retrieved = [(c.section_path or "", c.content) for c in found]
                except Exception as exc:  # noqa: BLE001
                    retrieved = [("retrieval failed", str(exc))]
            full = False
            while True:
                show_candidate(candidate, number, len(pending), chunks, retrieved, full=full)
                try:
                    choice = await asyncio.to_thread(
                        input, "\n[a]ccept  [e]dit  [s]kip  [d]etails  [q]uit > "
                    )
                    choice = choice.strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "q"
                if choice == "d":
                    full = True
                    continue
                if choice in ("a", "e"):
                    try:
                        edits = edit_candidate(candidate, source_text) if choice == "e" else None
                        record = accept_candidate(candidate, edits=edits)
                    except ValueError as exc:
                        out(f"  not accepted: {exc}")
                        break
                    accepted += 1
                    out(f"  accepted as {record.id}")
                    break
                if choice == "s":
                    record_skip(candidate.id)
                    skipped += 1
                    break
                if choice == "q":
                    raise SystemExit(0)
                out("  please type a, e, s, d or q")
    except SystemExit:
        pass
    finally:
        await redis.aclose()
        await engine.dispose()
        out(f"\nSession: accepted {accepted}, skipped {skipped}.")
        out("verified so far: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in progress().items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
