"""Run the RAG evaluation on the human-verified golden set.

    python -m eval.rag.run_eval --configs baseline_vector_only hybrid_no_rerank hybrid_rerank_local
    python -m eval.rag.run_eval --gate     # default config vs thresholds.yaml (exit 1 on fail)
    python -m eval.rag.run_eval --configs hybrid_no_rerank --no-ragas --limit 10   # quick, cheap

Every question is answered with use_cache=False. Retrieval metrics come from
RetrievalService.search(top_n=10); the answer, citations, latency and tokens come from answer().
RAGAS runs in a separate virtualenv (see ragas_scorer.py).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings, has_real_key
from app.core.db import create_engine
from app.core.llm import get_anthropic_chat_model, get_chat_model, get_embeddings
from app.core.logging import setup_logging
from app.models import PolicyChunk
from app.observability.pricing import PRICING_PATH
from app.observability.tracing import flush_tracing, init_tracing, run_trace, trace_id_for
from app.observability.usage import disable_tracking
from app.rag.answerer import AnswerGenerationError
from app.rag.config import RagSettings
from app.rag.reranker import build_reranker
from app.rag.service import RetrievalService
from eval import langfuse_push as lf
from eval.rag import metrics as M
from eval.rag.common import (
    CONFIGS_DIR,
    EVAL_DIR,
    QUESTIONS_PATH,
    RESULTS_DIR,
    THRESHOLDS_PATH,
    read_jsonl,
)
from eval.rag.schemas import MIN_VERIFIED_QUESTIONS, GoldenQuestion

logger = logging.getLogger("eval.run_eval")

RETRIEVAL_DEPTH = 10  # search(top_n=10): enough for hit@5 / MRR
RAGAS_PYTHON = os.environ.get("RAGAS_PYTHON", "/opt/ragas-venv/bin/python")
NO_RAGAS_CATEGORIES = ("unanswerable", "injection")  # faithfulness/precision make no sense there


# ----------------------------------------------------------------------------- config / data


@dataclass
class EvalConfig:
    name: str
    description: str
    settings: dict[str, Any] = field(default_factory=dict)


def load_config(
    name: str, configs_dir: Path = CONFIGS_DIR, _seen: tuple[str, ...] = ()
) -> EvalConfig:
    """configs/<name>.yaml; `extends: other` inherits its settings (child values win)."""
    if name in _seen:
        raise ValueError(f"config {name!r} extends itself: {' -> '.join((*_seen, name))}")
    path = configs_dir / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in configs_dir.glob("*.yaml")))
        raise FileNotFoundError(f"no config {name!r} in {configs_dir} (available: {available})")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings: dict[str, Any] = {}
    if raw.get("extends"):
        settings.update(load_config(raw["extends"], configs_dir, (*_seen, name)).settings)
    settings.update(raw.get("settings") or {})
    unknown = set(settings) - set(RagSettings.model_fields)
    if unknown:
        raise ValueError(f"config {name!r}: unknown RagSettings field(s) {sorted(unknown)}")
    return EvalConfig(
        name=raw.get("name", name), description=raw.get("description", ""), settings=settings
    )


def load_pricing(path: Path = PRICING_PATH) -> dict[str, dict[str, float]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["models"]


def load_verified_questions(path: Path = QUESTIONS_PATH) -> list[GoldenQuestion]:
    return [q for q in (GoldenQuestion(**r) for r in read_jsonl(path)) if q.verified_by_human]


def check_enough_questions(count: int, minimum: int = MIN_VERIFIED_QUESTIONS) -> str | None:
    """A message if the golden set is too small to evaluate on, else None."""
    if count >= minimum:
        return None
    return (
        f"Only {count} human-verified question(s) in golden/questions.jsonl; at least {minimum} "
        "are needed for numbers worth quoting. Review more candidates: `make eval-review`."
    )


# ----------------------------------------------------------------------------- token tracking


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens per model for EVERY chat call (answer, rewrite, multi-query)."""

    def __init__(self) -> None:
        self.by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            message = response.generations[0][0].message  # type: ignore[union-attr]
        except (IndexError, AttributeError):
            return
        usage = dict(getattr(message, "usage_metadata", None) or {})
        meta = getattr(message, "response_metadata", None) or {}
        model = meta.get("model_name") or meta.get("model") or "unknown"
        self.by_model[model][0] += usage.get("input_tokens", 0)
        self.by_model[model][1] += usage.get("output_tokens", 0)

    def snapshot(self) -> dict[str, tuple[int, int]]:
        return {m: (i, o) for m, (i, o) in self.by_model.items()}


def usage_since(
    before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]
) -> dict[str, tuple[int, int]]:
    out = {}
    for model, (inp, outp) in after.items():
        b_in, b_out = before.get(model, (0, 0))
        if inp - b_in or outp - b_out:
            out[model] = (inp - b_in, outp - b_out)
    return out


# ----------------------------------------------------------------------------- one question


def hits_of(chunks: list[Any]) -> list[M.Hit]:
    return [M.Hit(str(c.chunk_id), c.section_path, c.content) for c in chunks]


TRACE_PREFIX = ""  # set per run in main(): trace ids are derived from it, so scores can find them


async def evaluate_question(
    service: Any,
    tracker: UsageTracker,
    matcher: M.ReferenceMatcher,
    question: GoldenQuestion,
    pricing: dict[str, dict[str, float]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "reference_answer": question.reference_answer,
        "error": None,
    }
    trace_seed = f"{TRACE_PREFIX}{question.id}"
    record["trace_id"] = trace_id_for(trace_seed)
    try:
        with run_trace(
            "policy_query",
            seed=trace_seed,
            tags=["scenario:eval", f"category:{question.category}"],
            input={"query": question.question},
        ) as trace:
            retrieved = await service.search(
                question.question, question.policy_codes, top_n=RETRIEVAL_DEPTH
            )
            before = tracker.snapshot()
            started = time.perf_counter()
            result = await service.answer(question.question, question.policy_codes, use_cache=False)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            tokens = usage_since(before, tracker.snapshot())
            trace.update(
                output={"model": result.model_used, "insufficient": result.insufficient_context}
            )
    except (AnswerGenerationError, Exception) as exc:  # noqa: BLE001 - a failed question is data
        logger.warning("question_failed", extra={"id": question.id}, exc_info=True)
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return record

    retrieved_hits = hits_of(retrieved)
    used_hits = hits_of(result.chunks_used)
    citation_hits = [M.Hit(str(c.chunk_id), c.section_path) for c in result.citations]
    record.update(
        answer=result.answer,
        insufficient_context=result.insufficient_context,
        confidence=result.confidence,
        model_used=result.model_used,
        warnings=result.warnings,
        timings_ms=result.timings_ms,
        latency_ms=latency_ms,
        input_tokens=sum(i for i, _ in tokens.values()),
        output_tokens=sum(o for _, o in tokens.values()),
        cost_usd=sum(M.estimate_cost(m, i, o, pricing) for m, (i, o) in tokens.items()),
        citations=[
            {
                "chunk_id": str(c.chunk_id),
                "section_path": c.section_path,
                "pages": [c.page_start, c.page_end],
            }
            for c in result.citations
        ],
        retrieved=[
            {
                "rank": rank,
                "chunk_id": str(c.chunk_id),
                "section_path": c.section_path,
                "vector_rank": c.vector_rank,
                "keyword_rank": c.keyword_rank,
                "fused_score": c.fused_score,
                "rerank_score": c.rerank_score,
            }
            for rank, c in enumerate(retrieved, start=1)
        ],
        used=[
            {"chunk_id": str(c.chunk_id), "section_path": c.section_path}
            for c in result.chunks_used
        ],
        contexts_used=[c.content for c in result.chunks_used],
        hit_at_1=M.hit_at_k(matcher, question, retrieved_hits, 1),
        hit_at_3=M.hit_at_k(matcher, question, retrieved_hits, 3),
        hit_at_5=M.hit_at_k(matcher, question, retrieved_hits, 5),
        rr=M.reciprocal_rank(matcher, question, retrieved_hits),
        context_recall_sections=M.context_recall_sections(
            matcher, question, used_hits
        ),  # what the model was actually given (top_n)
        must_contain_ok=M.must_contain_ok(result.answer, question.must_contain),
        unanswerable_ok=(
            M.unanswerable_ok(result.insufficient_context, question.expect_insufficient_context)
            if question.category == "unanswerable"
            else None
        ),
        citation_ok=M.citation_ok(matcher, question, citation_hits, result.insufficient_context),
        injection_ok=M.injection_resisted(result.answer, question.forbidden_phrases),
    )
    return record


def finalize_record(
    record: dict[str, Any], question: GoldenQuestion, matcher: M.ReferenceMatcher
) -> None:
    """Decide failure class once all scores (incl. RAGAS faithfulness) are known."""
    faithfulness = record.get("faithfulness")
    failed = bool(
        record.get("error")
        or record.get("must_contain_ok") is False
        or record.get("unanswerable_ok") is False
        or record.get("injection_ok") is False
        or record.get("citation_ok") is False
        or (faithfulness is not None and faithfulness < 0.5)
    )
    # the chunks the model was actually given (answer()'s chunks_used), not search()'s top 10
    used = [
        M.Hit(c["chunk_id"], c["section_path"], text)
        for c, text in zip(record.get("used", []), record.get("contexts_used", []), strict=False)
    ]
    record["failed"] = failed
    record["failure_class"] = M.classify_failure(matcher, question, used, failed)


# ----------------------------------------------------------------------------- RAGAS


def build_ragas_rows(
    records: list[dict[str, Any]], questions: dict[str, GoldenQuestion]
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if (
            record["category"] in NO_RAGAS_CATEGORIES
            or record.get("error")
            or not record.get("answer")
            or not record.get("contexts_used")
        ):
            continue
        rows.append(
            {
                "id": record["id"],
                "user_input": record["question"],
                "response": record["answer"],
                "retrieved_contexts": record["contexts_used"],
                "reference": questions[record["id"]].reference_answer,
            }
        )
    return rows


@dataclass
class RagasOutcome:
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _write_ragas_input(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    work = RESULTS_DIR / ".tmp"
    work.mkdir(parents=True, exist_ok=True)
    stamp = f"{os.getpid()}_{int(time.time() * 1000)}"
    in_path, out_path = work / f"ragas_in_{stamp}.jsonl", work / f"ragas_out_{stamp}.jsonl"
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    in_path.write_text("\n".join(lines), encoding="utf-8")
    return in_path, out_path


def _read_ragas_output(in_path: Path, out_path: Path, ok: bool) -> list[dict[str, Any]] | None:
    """The worker's rows (None if it failed); always removes the temp files."""
    try:
        if not ok or not out_path.exists():
            return None
        return [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines() if x]
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


async def score_with_ragas(rows: list[dict[str, Any]], judge_model: str) -> RagasOutcome:
    if not rows:
        return RagasOutcome()
    python = Path(RAGAS_PYTHON)
    if not await asyncio.to_thread(python.exists):
        return RagasOutcome(
            error=f"{RAGAS_PYTHON} not found: build the image with INSTALL_EVAL=true"
        )
    in_path, out_path = await asyncio.to_thread(_write_ragas_input, rows)
    env = {
        **os.environ,
        "EVAL_JUDGE_MODEL": judge_model,
        "PYTHONPATH": str(EVAL_DIR.parent.parent),  # the dir that contains the `eval` package
        "PYTHONWARNINGS": "ignore",
    }
    process = await asyncio.create_subprocess_exec(
        str(python), "-m", "eval.rag.ragas_scorer", str(in_path), str(out_path),
        env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    _, stderr = await process.communicate()
    lines = await asyncio.to_thread(_read_ragas_output, in_path, out_path, process.returncode == 0)
    if lines is None:
        tail = stderr.decode(errors="replace")[-600:]
        return RagasOutcome(error=f"ragas worker failed (exit {process.returncode}): {tail}")
    summary = next((x["_summary"] for x in lines if "_summary" in x), {})
    return RagasOutcome(scores={x["id"]: x for x in lines if "id" in x}, summary=summary)


def merge_ragas(records: list[dict[str, Any]], outcome: RagasOutcome) -> None:
    for record in records:
        scored = outcome.scores.get(record["id"])
        if scored is None:
            continue
        record["faithfulness"] = scored.get("faithfulness")
        record["answer_relevancy"] = scored.get("answer_relevancy")
        record["context_precision"] = scored.get("context_precision")
        record["ragas_context_recall"] = scored.get("context_recall")
        if scored.get("errors"):
            record["ragas_errors"] = scored["errors"]


# ----------------------------------------------------------------------------- run one config


async def valid_chunk_ids(session_factory: async_sessionmaker[AsyncSession]) -> set[str]:
    async with session_factory() as session:
        return {str(i) for i in (await session.execute(select(PolicyChunk.id))).scalars()}


def build_service(
    config: EvalConfig,
    session_factory: async_sessionmaker[AsyncSession],
    tracker: UsageTracker,
    provider: str = "openai",
) -> RetrievalService:
    rag = RagSettings(**{**config.settings, "cache_enabled": False})
    if provider == "anthropic":  # the evaluation calls Claude (ANTHROPIC_API_KEY)
        chat_factory = lambda: get_anthropic_chat_model(callbacks=[tracker])  # noqa: E731
    else:
        chat_factory = lambda: get_chat_model(temperature=0, callbacks=[tracker])  # noqa: E731
    return RetrievalService(
        session_factory=session_factory,
        embeddings=get_embeddings(),
        chat_model_factory=chat_factory,
        reranker=build_reranker(rag),
        cache=None,  # evaluation never uses the semantic cache
        settings=rag,
    )


async def run_config(
    config: EvalConfig,
    questions: list[GoldenQuestion],
    service: Any,
    tracker: UsageTracker,
    matcher: M.ReferenceMatcher,
    pricing: dict[str, dict[str, float]],
    *,
    judge_model: str | None,
    progress: bool = True,
) -> dict[str, Any]:
    """Questions run one after another (clean latency numbers). Returns records + aggregates."""
    records: list[dict[str, Any]] = []
    for number, question in enumerate(questions, start=1):
        record = await evaluate_question(service, tracker, matcher, question, pricing)
        record["config"] = config.name
        records.append(record)
        if progress:
            status = "ERROR" if record.get("error") else f"{record.get('latency_ms', 0):.0f} ms"
            sys.stdout.write(
                f"  [{config.name}] {number}/{len(questions)} {question.id} {status}\n"
            )

    by_id = {q.id: q for q in questions}
    ragas = RagasOutcome()
    if judge_model:
        rows = build_ragas_rows(records, by_id)
        sys.stdout.write(
            f"  [{config.name}] RAGAS judge={judge_model} on {len(rows)} answers ...\n"
        )
        ragas = await score_with_ragas(rows, judge_model)
        if ragas.error:
            sys.stdout.write(f"  RAGAS failed: {ragas.error}\n")
        merge_ragas(records, ragas)
    for record in records:
        finalize_record(record, by_id[record["id"]], matcher)

    aggregated = M.aggregate(records)
    aggregated["by_category"] = M.aggregate_by_category(records)
    if ragas.summary:
        judge_cost = M.estimate_cost(
            judge_model,
            ragas.summary.get("judge_input_tokens", 0),
            ragas.summary.get("judge_output_tokens", 0),
            pricing,
        )
        aggregated["judge"] = {**ragas.summary, "cost_usd": judge_cost}
    if ragas.error:
        aggregated["ragas_error"] = ragas.error
    return {
        "config": {
            "name": config.name,
            "description": config.description,
            "settings": config.settings,
        },
        "records": records,
        "summary": aggregated,
    }


# ----------------------------------------------------------------------------- gate


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


GATE_METRIC_NAMES = {
    "hit_at_5": "hit_at_5",
    "faithfulness": "faithfulness",
    "must_contain_rate": "must_contain_rate",
    "unanswerable_accuracy": "unanswerable_accuracy",
}


def evaluate_gate(
    metrics: dict[str, float | None], thresholds: dict[str, float], *, ragas_enabled: bool
) -> list[str]:
    """Failure messages (empty = pass). A missing metric fails, except faithfulness when RAGAS
    was switched off on purpose."""
    failures = []
    for key, minimum in thresholds.items():
        name = GATE_METRIC_NAMES.get(key, key)
        value = metrics.get(name)
        if value is None:
            if name == "faithfulness" and not ragas_enabled:
                continue
            failures.append(f"{key}: no value (threshold {minimum})")
        elif value < minimum:
            failures.append(f"{key}: {value:.3f} < {minimum}")
    return failures


# ----------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m eval.rag.run_eval", description=__doc__)
    parser.add_argument("--configs", nargs="+", help="config names in eval/rag/configs/")
    parser.add_argument("--gate", action="store_true", help="default config vs thresholds.yaml")
    parser.add_argument("--judge", help="RAGAS judge model (default $EVAL_JUDGE_MODEL or gpt-4o)")
    parser.add_argument("--no-ragas", action="store_true", help="skip the LLM-judged RAGAS metrics")
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="push scores, the golden dataset and a dataset run to Langfuse (needs its keys)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        help="generator LLM (default: anthropic when ANTHROPIC_API_KEY is set, else openai)",
    )
    parser.add_argument("--limit", type=int, help="only the first N verified questions")
    parser.add_argument("--name", help="run folder suffix")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--database-url", help="evaluate against another database (experiment DB)")
    return parser.parse_args(argv)


def push_rag_to_langfuse(
    run_name: str, stamp: str, questions: list[GoldenQuestion], results: list[dict[str, Any]]
) -> None:
    """Golden question dataset + one dataset run per config + scores on each question trace."""
    if not lf.ready():
        sys.stdout.write("--langfuse given but no Langfuse keys are configured: nothing pushed\n")
        return
    items = [
        {
            "id": q.id,
            "input": {"question": q.question, "policy_codes": q.policy_codes},
            "expected_output": {"reference_answer": q.reference_answer},
            "metadata": {"category": q.category},
        }
        for q in questions
    ]
    lf.ensure_dataset(lf.RAG_DATASET, "Verified golden questions (ClaimPilot RAG eval)", items)
    for result in results:
        config = result["config"]["name"]
        langfuse_run = f"{config}-{stamp}"
        for record in result["records"]:
            if not record.get("trace_id") or record.get("error"):
                continue
            lf.link_run(langfuse_run, record["id"], record["trace_id"], {"config": config})
            lf.push_scores(
                record["trace_id"],
                {
                    "faithfulness": record.get("faithfulness"),
                    "hit_at_3": record.get("hit_at_3"),
                    "citation_correct": record.get("citation_ok"),
                },
                comment=langfuse_run,
            )
        sys.stdout.write(f"  [{config}] pushed scores to Langfuse as run {langfuse_run}\n")
    lf.finish()


async def main(argv: list[str] | None = None) -> int:
    from eval.rag.report import write_report  # local import: report needs no app deps

    args = parse_args(argv)
    questions = load_verified_questions(args.questions)
    problem = check_enough_questions(len(questions))
    if problem:
        sys.stderr.write(problem + "\n")
        return 2

    thresholds = load_thresholds() if args.gate else None
    config_names = args.configs or ([thresholds["default_config"]] if thresholds else None)
    if not config_names:
        sys.stderr.write("Pass --configs NAME [NAME ...] or --gate.\n")
        return 2
    configs = [load_config(n) for n in config_names]
    if args.limit:
        questions = questions[: args.limit]

    judge_model = None if args.no_ragas else (
        args.judge or os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o")
    )  # fmt: skip
    settings = get_settings()
    if judge_model and judge_model == settings.primary_model:
        sys.stdout.write(
            f"WARNING: judge model == generator model ({judge_model}); scores may be inflated.\n"
        )

    setup_logging("WARNING")
    disable_tracking()  # evaluation spend must not use up the live daily LLM budget
    init_tracing(settings)  # eval questions are traced as policy_query (scenario:eval)
    engine = create_engine(args.database_url or settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    pricing = load_pricing()
    started = datetime.now()
    run_name = args.name or "+".join(c.name for c in configs)
    run_dir = args.results_dir / f"{started:%Y%m%d-%H%M%S}_{run_name[:60]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    provider = args.provider or (
        "anthropic" if has_real_key(settings.anthropic_api_key) else "openai"
    )
    results: list[dict[str, Any]] = []
    try:
        matcher = M.ReferenceMatcher(await valid_chunk_ids(session_factory))
        for config in configs:
            sys.stdout.write(f"\n== {config.name}: {config.description}\n")
            tracker = UsageTracker()
            global TRACE_PREFIX
            TRACE_PREFIX = f"evalrag-{started:%Y%m%d%H%M%S}-{config.name}-"
            service = build_service(config, session_factory, tracker, provider=provider)
            results.append(
                await run_config(
                    config, questions, service, tracker, matcher, pricing, judge_model=judge_model
                )
            )
    finally:
        await engine.dispose()

    if args.langfuse:
        push_rag_to_langfuse(run_name, f"{started:%Y%m%d-%H%M%S}", questions, results)
    flush_tracing()
    with (run_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            for record in result["records"]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "run": run_name,
        "started": started.isoformat(timespec="seconds"),
        "n_questions": len(questions),
        "judge_model": judge_model,
        "generator_model": settings.primary_model,
        "configs": {r["config"]["name"]: r["summary"] | {"config": r["config"]} for r in results},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(run_dir, summary, [r for res in results for r in res["records"]])
    sys.stdout.write(f"\nResults: {run_dir}\n  report: {run_dir / 'report.md'}\n")

    if thresholds:
        name = configs[0].name
        failures = evaluate_gate(
            summary["configs"][name]["metrics"],
            thresholds["thresholds"],
            ragas_enabled=judge_model is not None,
        )
        if failures:
            sys.stdout.write(f"\nGATE FAILED for {name}:\n  " + "\n  ".join(failures) + "\n")
            return 1
        sys.stdout.write(f"\nGATE PASSED for {name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
