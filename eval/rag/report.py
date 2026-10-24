"""Markdown + CSV report from the per-config summaries and per-question records.

No API calls and no app imports: it only reads the dicts produced by run_eval, so it can be
unit-tested from a fake summary.
"""

import csv
from pathlib import Path
from typing import Any

from eval.rag.metrics import worst_questions

# (metric key, column header)
COMPARISON_COLUMNS = [
    ("hit_at_3", "hit@3"),
    ("mrr", "MRR"),
    ("context_recall_sections", "ctx recall (sections)"),
    ("faithfulness", "faithfulness"),
    ("answer_relevancy", "answer relevancy"),
    ("context_precision", "ctx precision"),
    ("must_contain_rate", "must_contain"),
    ("unanswerable_accuracy", "unanswerable acc"),
]
QUALITY_METRICS = ("hit_at_3", "faithfulness", "must_contain_rate", "unanswerable_accuracy")
NOISE = 0.05  # one question out of 20 is 0.05: smaller gaps are not evidence

# (base config, variant config, label) pairs the findings section looks for
COMPARISONS = [
    ("baseline_vector_only", "hybrid_no_rerank", "Adding keyword search (hybrid RRF)"),
    ("hybrid_no_rerank", "hybrid_rerank_local", "Adding the local reranker"),
    ("hybrid_rerank_local", "hybrid_rerank_multiquery", "Adding multi-query"),
    ("hybrid_rerank_local", "topn_3", "Sending 3 clauses to the model instead of 5"),
    ("hybrid_rerank_local", "topn_8", "Sending 8 clauses to the model instead of 5"),
]


def fmt(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def metric(summary: dict[str, Any], config: str, key: str) -> float | None:
    return summary["configs"][config]["metrics"].get(key)


def p95(summary: dict[str, Any], config: str) -> float | None:
    return summary["configs"][config]["latency_ms"]["p95"]


def cost_per_question(config_summary: dict[str, Any]) -> float | None:
    generator = config_summary.get("avg_generator_cost_usd")
    judge = (config_summary.get("judge") or {}).get("cost_usd")
    if generator is None:
        return None
    return generator + (judge / config_summary["n"] if judge and config_summary["n"] else 0.0)


def quality_score(config_summary: dict[str, Any]) -> float:
    values = [config_summary["metrics"].get(k) for k in QUALITY_METRICS]
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else 0.0


def best_config(summary: dict[str, Any]) -> str:
    """Highest mean of hit@3, faithfulness, must_contain_rate, unanswerable_accuracy;
    ties go to the lower p95 latency."""
    return max(
        summary["configs"],
        key=lambda c: (
            round(quality_score(summary["configs"][c]), 4),
            -(summary["configs"][c]["latency_ms"]["p95"] or float("inf")),
        ),
    )


def comparison_table(summary: dict[str, Any]) -> str:
    headers = ["config", *[h for _, h in COMPARISON_COLUMNS], "p95 latency (ms)", "cost / question"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for name, cfg in summary["configs"].items():
        row = [name]
        row += [fmt(cfg["metrics"].get(k)) for k, _ in COMPARISON_COLUMNS]
        row += [ms(cfg["latency_ms"]["p95"]), usd(cost_per_question(cfg))]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def category_table(config_summary: dict[str, Any]) -> str:
    headers = ["category", "n", "hit@3", "must_contain", "faithfulness", "answer relevancy",
               "citation validity", "unanswerable acc", "injection resisted"]  # fmt: skip
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for category, cat in config_summary["by_category"].items():
        m = cat["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    str(cat["n"]),
                    fmt(m.get("hit_at_3")),
                    fmt(m.get("must_contain_rate")),
                    fmt(m.get("faithfulness")),
                    fmt(m.get("answer_relevancy")),
                    fmt(m.get("citation_validity")),
                    fmt(m.get("unanswerable_accuracy")),
                    fmt(m.get("injection_resistance")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def what_went_wrong(record: dict[str, Any]) -> list[str]:
    reasons = []
    if record.get("error"):
        reasons.append(f"error: {record['error']}")
    if record.get("hit_at_5") is False:
        reasons.append("reference clause not in the top 5 retrieved")
    elif record.get("hit_at_5") and record.get("failure_class") == "retrieval_miss":
        reasons.append("reference was retrieved but not among the clauses sent to the model")
    if record.get("must_contain_ok") is False:
        reasons.append("answer is missing a required fact (must_contain)")
    if record.get("unanswerable_ok") is False:
        reasons.append("wrong refusal behaviour (answered an unanswerable question or refused one "
                       "that is answerable)")  # fmt: skip
    if record.get("injection_ok") is False:
        reasons.append("followed the injected instruction")
    if record.get("citation_ok") is False:
        reasons.append("citations do not point to a reference clause")
    faithfulness = record.get("faithfulness")
    if faithfulness is not None and faithfulness < 0.8:
        reasons.append(
            f"low faithfulness ({faithfulness:.2f}): claims not supported by the clauses"
        )
    return reasons or ["no single failing check (lowest overall score)"]


def worst_section(records: list[dict[str, Any]], limit: int = 5) -> str:
    worst = worst_questions(records, limit)
    if not worst:
        return "No question failed any check."
    blocks = []
    for number, record in enumerate(worst, start=1):
        retrieved = record.get("retrieved", [])[:3]
        sections = (
            "; ".join(f"{r['rank']}. {(r['section_path'] or '')[:60]}" for r in retrieved) or "n/a"
        )
        cited = "; ".join(c["section_path"] or "" for c in record.get("citations", [])) or "none"
        kind = record.get("failure_class", "none").replace("_", " ")
        blocks.append(
            f"**{number}. {record['id']}** ({record['category']}) - *{kind}*\n"
            f"- Question: {record['question']}\n"
            f"- Expected: {record['reference_answer'][:200]}\n"
            f"- Got: {(record.get('answer') or '')[:300]}\n"
            f"- Top retrieved: {sections}\n"
            f"- Cited: {cited}\n"
            f"- What went wrong: {'; '.join(what_went_wrong(record))}"
        )
    return "\n\n".join(blocks)


def delta_phrase(label: str, base: float | None, new: float | None) -> str | None:
    if base is None or new is None:
        return None
    gap = new - base
    verdict = "within noise" if abs(gap) < NOISE else ("better" if gap > 0 else "worse")
    return f"{label} {fmt(base)} -> {fmt(new)} ({gap:+.2f}, {verdict})"


def findings(summary: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    configs = summary["configs"]
    n = summary["n_questions"]
    out: list[str] = []
    best = best_config(summary)
    out.append(
        f"Best config by quality score (mean of hit@3, faithfulness, must_contain, unanswerable "
        f"accuracy): **{best}** ({quality_score(configs[best]):.2f}), p95 "
        f"{ms(p95(summary, best))} ms. Ties are broken by lower p95 latency."
    )
    if len(configs) > 1:
        ranking = sorted(configs, key=lambda c: -quality_score(configs[c]))
        out.append(
            "Quality ranking: "
            + " > ".join(f"{c} ({quality_score(configs[c]):.2f})" for c in ranking)
            + f". With n={n}, differences below {NOISE:.2f} are not evidence."
        )
    for base, variant, label in COMPARISONS:
        if base not in configs or variant not in configs:
            continue
        parts = [
            delta_phrase(name, metric(summary, base, key), metric(summary, variant, key))
            for key, name in (
                ("hit_at_3", "hit@3"),
                ("context_precision", "context precision"),
                ("faithfulness", "faithfulness"),
                ("must_contain_rate", "must_contain"),
                ("unanswerable_accuracy", "unanswerable acc"),
            )
        ]
        latency_gap = (p95(summary, variant) or 0) - (p95(summary, base) or 0)
        parts.append(f"p95 latency {ms(p95(summary, base))} -> {ms(p95(summary, variant))} ms "
                     f"({latency_gap:+,.0f} ms)")  # fmt: skip
        out.append(f"{label} ({base} -> {variant}): " + "; ".join(p for p in parts if p) + ".")

    best_records = [r for r in records if r.get("config") == best]
    failures = [r for r in best_records if r.get("failed")]
    if failures:
        retrieval = sum(1 for r in failures if r["failure_class"] == "retrieval_miss")
        out.append(
            f"For {best}, {len(failures)} of {len(best_records)} questions failed a check: "
            f"{retrieval} retrieval misses and {len(failures) - retrieval} generation issues."
        )
    errors = sum(c["errors"] for c in configs.values())
    if errors:
        out.append(f"{errors} question run(s) ended in an error; they count as failures.")
    for name, cfg in configs.items():
        if cfg.get("ragas_error"):
            out.append(f"RAGAS did not run for {name}: {cfg['ragas_error'][:200]}")
    if summary.get("judge_model") and summary["judge_model"] == summary.get("generator_model"):
        out.append("The judge model equals the generator model: RAGAS scores may be inflated.")
    return out


def write_csv(run_dir: Path, summary: dict[str, Any]) -> None:
    extra = ["hit_at_1", "hit_at_5", "ragas_context_recall", "citation_validity"]
    keys = [k for k, _ in COMPARISON_COLUMNS] + extra + ["injection_resistance"]
    with (run_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "n", *keys, "p50_ms", "p95_ms", "avg_input_tokens",
                         "avg_output_tokens", "generator_cost_usd", "judge_cost_usd",
                         "cost_per_question_usd"])  # fmt: skip
        for name, cfg in summary["configs"].items():
            judge_cost = (cfg.get("judge") or {}).get("cost_usd")
            writer.writerow(
                [name, cfg["n"], *[cfg["metrics"].get(k) for k in keys],
                 cfg["latency_ms"]["p50"], cfg["latency_ms"]["p95"], cfg["avg_input_tokens"],
                 cfg["avg_output_tokens"], cfg["avg_generator_cost_usd"], judge_cost,
                 cost_per_question(cfg)]
            )  # fmt: skip


def stage_table(config_summary: dict[str, Any]) -> str:
    lines = ["| stage | p50 (ms) | p95 (ms) |", "|---|---|---|"]
    for stage, values in config_summary["stage_latency_ms"].items():
        lines.append(f"| {stage} | {ms(values['p50'])} | {ms(values['p95'])} |")
    return "\n".join(lines)


def write_report(run_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    best = best_config(summary)
    best_summary = summary["configs"][best]
    best_records = [r for r in records if r.get("config") == best]
    judge = summary.get("judge_model") or "none (RAGAS skipped)"

    parts = [
        f"# RAG evaluation: {summary['run']}",
        "",
        f"- Run started: {summary['started']}",
        f"- Questions: {summary['n_questions']} human-verified (golden/questions.jsonl)",
        f"- Generator: {summary['generator_model']} (answers recorded with the model that actually "
        f"answered); RAGAS judge: {judge}",
        f"- Configs: {', '.join(summary['configs'])}",
        "- Retrieval metrics use search(top_n=10); answers use answer(use_cache=False).",
        "",
        "## Comparison",
        "",
        comparison_table(summary),
        "",
        "Cost / question = generator tokens (answer, rewrite, multi-query) + RAGAS judge share, at "
        "the prices in eval/rag/pricing.yaml (assumptions: edit them). "
        "ctx recall (sections) = share of reference clauses among the clauses given to the model.",
        "",
        f"## Best config: {best}",
        "",
        "### Per category",
        "",
        category_table(best_summary),
        "",
        "### Latency by stage",
        "",
        stage_table(best_summary),
        "",
        "### Models that answered",
        "",
        ", ".join(f"{m}: {c}" for m, c in best_summary["models"].items()),
        "",
        f"### 5 worst questions ({best})",
        "",
        worst_section(best_records),
        "",
        "## Findings",
        "",
        *[f"- {line}" for line in findings(summary, records)],
        "",
    ]
    path = run_dir / "report.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    write_csv(run_dir, summary)
    return path
