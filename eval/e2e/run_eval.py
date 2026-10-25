"""End-to-end evaluation of the full claim workflow (Module 9).

    python -m eval.e2e.run_eval                                   # default config, all claims
    python -m eval.e2e.run_eval --configs default deterministic_routing --repeat 3
    python -m eval.e2e.run_eval --gate                            # exit 1 below thresholds.yaml
    python -m eval.e2e.run_eval --claims CLM-014 ADV-001 --skip-setup   # quick check

Runs against EVAL_DATABASE_URL only (default: <DATABASE_URL database>_eval), never the main one.
Each run stops at the human-approval interrupt; nothing is approved (one adversarial excepted).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.observability.tracing import flush_tracing, init_tracing, trace_id_for
from app.observability.usage import disable_tracking
from eval import langfuse_push as lf
from eval.e2e import metrics as M
from eval.e2e.attribution import attribute, is_failed
from eval.e2e.common import (
    RESULTS_DIR,
    RULE_MAP_PATH,
    THRESHOLDS_PATH,
    WORK_DIR,
    Case,
    EvalConfig,
    assert_eval_database,
    eval_database_url,
    load_cases,
    load_config,
    load_yaml,
)
from eval.e2e.harness import EvalEnv, judge_supported
from eval.e2e.report import render_compare, write_report
from eval.e2e.setup_db import migrate, reset_state, setup, sync_wordings

logger = logging.getLogger("eval.e2e")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m eval.e2e.run_eval", description=__doc__)
    p.add_argument("--configs", nargs="+", default=["default"])
    p.add_argument("--repeat", type=int, default=1, help="runs per claim (decision stability)")
    p.add_argument("--claims", nargs="+", help="only these claim ids")
    p.add_argument("--limit", type=int, help="only the first N golden claims")
    p.add_argument("--no-adversarial", action="store_true")
    p.add_argument(
        "--langfuse",
        action="store_true",
        help="push scores, the golden dataset and a dataset run to Langfuse (needs its keys)",
    )
    p.add_argument("--skip-setup", action="store_true", help="reuse the existing eval database")
    p.add_argument(
        "--no-judge", action="store_true", help="do not call the LLM judge for citations"
    )
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("EVAL_CONCURRENCY", "3")))
    p.add_argument(
        "--timeout", type=float, default=float(os.environ.get("EVAL_RUN_TIMEOUT", "600"))
    )
    p.add_argument(
        "--rescore",
        metavar="RESULTS_DIR",
        help="re-score + re-attribute a finished run from its runs.jsonl (no LLM, no database)",
    )
    p.add_argument(
        "--gate", action="store_true", help="exit 1 if the first config misses thresholds.yaml"
    )
    return p.parse_args(argv)


def effective_rule_map(rule_map: dict[str, Any], cfg: EvalConfig) -> dict[str, Any]:
    """`wording_policies` = policies with a wording in THIS config; `real_wording_policies` = real
    PDFs only (the wording-available subset and the headline citation numbers use those)."""
    real = list(rule_map.get("wording_policies") or [])
    out = {**rule_map, "real_wording_policies": real}
    if cfg.synthetic_wording:
        out["wording_policies"] = real + list(rule_map.get("synthetic_policies") or [])
    return out


def push_to_langfuse(
    config: str,
    stamp: str,
    cases: list[Case],
    scores: list[dict[str, Any]],
    adversarial: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> None:
    """Golden dataset + one dataset run + scores on the trace of every case."""
    if not lf.ready():
        logger.warning("--langfuse given but no Langfuse keys are configured: nothing pushed")
        return
    items = [
        {
            "id": c.claim_id,
            "input": {"claim_id": c.claim_id, "scenario": c.scenario, "kind": c.kind},
            "expected_output": (c.adversarial or c.golden)
            if c.kind == "adversarial"
            else {
                "decision": c.golden.get("expected_decision"),
                "payable": c.golden.get("expected_payable_amount"),
            },
            "metadata": {"policy_id": c.golden.get("policy_id")},
        }
        for c in cases
    ]
    lf.ensure_dataset(lf.E2E_DATASET, "Golden + adversarial claims (ClaimPilot e2e eval)", items)
    run_name = f"{config}-{stamp}"
    by_claim = {s["claim_id"]: s for s in scores}
    adv = {a["claim_id"]: a for a in adversarial}
    for run in runs:
        if not run.get("run_id"):  # a run that crashed before it started has no trace
            continue
        trace_id = trace_id_for(run["run_id"])
        lf.link_run(run_name, run["claim_id"], trace_id, {"config": config})
        s, a = by_claim.get(run["claim_id"]), adv.get(run["claim_id"])
        values: dict[str, float | bool | None] = {}
        if s is not None:
            status = (s.get("citation") or {}).get("status")
            values = {
                "decision_correct": s["decision_correct"],
                "amount_correct": s.get("amount_ok"),
                "citation_correct": {"ok": 1.0, "bad": 0.0}.get(status),
            }
        elif a is not None:
            values = {"adversarial_pass": a["passed"]}
        lf.push_scores(trace_id, values, comment=run_name)
    lf.finish()
    logger.info("[%s] pushed %d traces to Langfuse as run %s", config, len(runs), run_name)


def select_cases(cases: list[Case], args: argparse.Namespace) -> list[Case]:
    if args.claims:
        wanted = set(args.claims)
        return [c for c in cases if c.claim_id in wanted]
    if args.limit:
        golden = [c for c in cases if c.kind == "golden"][: args.limit]
        return golden + [c for c in cases if c.kind == "adversarial"]
    return cases


async def run_pass(
    env: EvalEnv, cases: list[Case], cfg: EvalConfig, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Every case once, `concurrency` at a time."""
    gate = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(case: Case) -> dict[str, Any]:
        nonlocal done
        async with gate:
            run = await env.run_case(case, cfg, args.timeout)
        done += 1
        decision = (run.get("decision") or {}).get("decision")
        logger.info(
            "[%s] %d/%d %s -> %s (%.0fs)",
            cfg.name,
            done,
            len(cases),
            case.claim_id,
            decision or run["status"],
            run["latency_s"],
        )
        return run

    return list(await asyncio.gather(*(one(c) for c in cases)))


async def apply_judge(
    scores: list[dict[str, Any]], runs: dict[str, dict[str, Any]], enabled: bool
) -> None:
    """Citations whose section matched no keyword: an LLM judge decides (marked as judged)."""
    for s in scores:
        cite = s["citation"]
        if cite["status"] != "needs_judge":
            continue
        texts = [runs[s["claim_id"]]["clause_texts"].get(c, "") for c in cite["chunk_ids"]]
        if not enabled:
            s["citation"] = {"status": "skipped", "reason": "judge disabled"}  # not counted
            continue
        verdicts = [await judge_supported(cite["describes"], t) for t in texts if t]
        s["citation"] = {"status": "ok" if any(verdicts) else "bad", "judged": True}


def evaluate(
    cases: list[Case], runs: list[dict[str, Any]], rule_map: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(golden scores, adversarial scores, attribution) for one pass."""
    by_id = {r["claim_id"]: r for r in runs}
    scores, adversarial, attribution = [], [], []
    for case in cases:
        run = by_id[case.claim_id]
        if case.kind == "adversarial":
            extra = {"notification_messages": run.get("notification_messages") or []}
            adversarial.append(M.score_adversarial(case.adversarial, run, extra))
            continue
        score = M.score_claim(case.golden, run, rule_map)
        scores.append(score)
    return scores, adversarial, attribution


def attribute_all(
    cases: list[Case],
    scores: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    rule_map: dict[str, Any],
) -> list[dict[str, Any]]:
    by_case = {c.claim_id: c for c in cases}
    rows = []
    for s in scores:
        if not is_failed(s):
            continue
        case = by_case[s["claim_id"]]
        verdict = attribute(s, case.golden, case.truth, runs[s["claim_id"]], rule_map)
        if verdict:
            rows.append(
                {
                    "claim_id": s["claim_id"],
                    "scenario": f"{s['scenario_type']}/{s['subtype']}",
                    "expected": s["expected_decision"],
                    "predicted": s["predicted_decision"],
                    **verdict,
                }
            )
    return rows


def write_results(
    folder: Path,
    cfg: EvalConfig,
    runs: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    adversarial: list[dict[str, Any]],
    attribution: list[dict[str, Any]],
    stability: dict[str, Any],
    rule_map: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    wording = set(rule_map.get("real_wording_policies") or rule_map.get("wording_policies") or [])
    golden_runs = [r for r in runs if r.get("kind") == "golden"]
    subset_scores = [s for s in scores if s["policy_id"] in wording]
    subset_ids = {s["claim_id"] for s in subset_scores}
    metrics = {
        "all": M.aggregate(scores, golden_runs, adversarial),
        "wording": M.aggregate(
            subset_scores, [r for r in golden_runs if r["claim_id"] in subset_ids], []
        ),
    }
    metrics["wording"]["adversarial_pass_rate"] = metrics["all"]["adversarial_pass_rate"]
    metrics["wording"]["adversarial_n"] = metrics["all"]["adversarial_n"]
    (folder / "config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (folder / "scores.json").write_text(json.dumps(scores, indent=2, default=str), encoding="utf-8")
    (folder / "attribution.json").write_text(json.dumps(attribution, indent=2), encoding="utf-8")
    (folder / "adversarial.json").write_text(json.dumps(adversarial, indent=2), encoding="utf-8")
    (folder / "stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (folder / "runs.jsonl").open("w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run, default=str) + "\n")
    write_report(folder)


def check_gate(metrics: dict[str, Any]) -> list[str]:
    spec = load_yaml(THRESHOLDS_PATH)
    failures = []
    for name, minimum in (spec.get("thresholds") or {}).items():
        value = metrics.get(name)
        if value is None or value < minimum:
            failures.append(f"{name} = {value} < {minimum}")
    for name, maximum in (spec.get("max_thresholds") or {}).items():
        value = metrics.get(name)
        if value is None or value > maximum:
            failures.append(f"{name} = {value} > {maximum}")
    return failures


def rescore(folder: Path, no_adversarial: bool = False) -> None:
    """Recompute scores, attribution, metrics and the report from a saved runs.jsonl.

    Citation judgements are not repeated: they are taken from the saved scores.json."""
    rule_map = load_yaml(RULE_MAP_PATH)
    cfg = load_config(json.loads((folder / "config.json").read_text())["name"])
    rule_map = effective_rule_map(rule_map, cfg)
    all_runs = [
        json.loads(line) for line in (folder / "runs.jsonl").read_text().splitlines() if line
    ]
    meta = json.loads((folder / "meta.json").read_text())
    first = [r for r in all_runs if r.get("repeat", 0) == 0]
    cases = [
        c
        for c in load_cases(include_adversarial=not no_adversarial)
        if c.claim_id in {r["claim_id"] for r in first}
    ]
    by_id = {r["claim_id"]: r for r in first}
    scores, adversarial, _ = evaluate(cases, first, rule_map)
    old = {s["claim_id"]: s["citation"] for s in json.loads((folder / "scores.json").read_text())}
    for s in scores:
        if s["citation"]["status"] == "needs_judge" and s["claim_id"] in old:
            s["citation"] = old[s["claim_id"]]
    decisions: dict[str, list[str]] = {}
    for run in all_runs:
        decisions.setdefault(run["claim_id"], []).append(
            (run.get("decision") or {}).get("decision") or run["status"]
        )
    attribution = attribute_all(cases, scores, by_id, rule_map)
    write_results(
        folder,
        cfg,
        all_runs,
        scores,
        adversarial,
        attribution,
        M.stability(decisions),
        rule_map,
        meta,
    )


async def amain(args: argparse.Namespace) -> int:
    if args.rescore:
        rescore(Path(args.rescore))
        return 0
    settings = get_settings()
    setup_logging(settings.log_level)
    for noisy in (
        "mcp",
        "httpx",
        "app.core.llm",
        "app.agents",
        "app.rag",
        "app.intake",
        "langchain_mcp_adapters",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("eval.e2e").setLevel(logging.INFO)
    disable_tracking()  # evaluation spend must not use up the live daily LLM budget
    init_tracing(settings)  # eval runs are traced (scenario:eval) when Langfuse keys are set
    main_url = settings.database_url
    eval_url = eval_database_url(main_url)
    assert_eval_database(eval_url, main_url)

    data_root = WORK_DIR / "data"
    if not args.skip_setup:
        logger.info("setting up the eval database %s", eval_url.rsplit("/", 1)[-1])
        data_root = await setup(eval_url, main_url)
    else:  # an existing eval database must still match the current schema
        await asyncio.to_thread(migrate, eval_url)

    rule_map = load_yaml(RULE_MAP_PATH)
    cases = select_cases(load_cases(include_adversarial=not args.no_adversarial), args)
    env = await EvalEnv.open(eval_url, data_root)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: list[dict[str, Any]] = []
    first_metrics: dict[str, Any] = {}
    try:
        for name in args.configs:
            cfg = load_config(name)
            cfg_map = effective_rule_map(rule_map, cfg)
            await sync_wordings(
                env.engine,
                eval_url,
                data_root,
                cfg_map["wording_policies"],
                list(rule_map.get("synthetic_policies") or []),
            )
            all_runs: list[dict[str, Any]] = []
            decisions: dict[str, list[str]] = {}
            first: tuple[list, list, list] | None = None
            for repeat in range(args.repeat):
                await reset_state(env.engine, drop_extractions=cfg.fresh_extraction)
                runs = await run_pass(env, cases, cfg, args)
                for run in runs:
                    run["repeat"] = repeat
                    decisions.setdefault(run["claim_id"], []).append(
                        (run.get("decision") or {}).get("decision") or run["status"]
                    )
                all_runs += runs
                if repeat == 0:
                    by_id = {r["claim_id"]: r for r in runs}
                    scores, adversarial, _ = evaluate(cases, runs, cfg_map)
                    await apply_judge(scores, by_id, not args.no_judge)
                    attribution = attribute_all(cases, scores, by_id, cfg_map)
                    first = (scores, adversarial, attribution)
            assert first is not None
            scores, adversarial, attribution = first
            folder = RESULTS_DIR / f"{stamp}_{cfg.name}"
            meta = {
                "started": datetime.now().isoformat(timespec="seconds"),
                "repeat": args.repeat,
                "database": eval_url.rsplit("/", 1)[-1],
            }
            write_results(
                folder,
                cfg,
                all_runs,
                scores,
                adversarial,
                attribution,
                M.stability(decisions),
                cfg_map,
                meta,
            )
            if args.langfuse:
                push_to_langfuse(
                    cfg.name,
                    stamp,
                    cases,
                    scores,
                    adversarial,
                    [r for r in all_runs if r.get("repeat", 0) == 0],
                )
            logger.info("[%s] report: %s", cfg.name, folder / "report.md")
            from eval.e2e.report import load_result

            results.append(load_result(folder))
            first_metrics = (
                first_metrics or json.loads((folder / "metrics.json").read_text())["all"]
            )
        if len(results) > 1:
            compare = RESULTS_DIR / f"{stamp}_compare"
            compare.mkdir(parents=True, exist_ok=True)
            (compare / "report.md").write_text(render_compare(results), encoding="utf-8")
            logger.info("comparison: %s", compare / "report.md")
    finally:
        await env.close()
        flush_tracing()

    if args.gate:
        failures = check_gate(first_metrics)
        for failure in failures:
            sys.stderr.write(f"GATE FAIL: {failure}\n")
        return 1 if failures else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(amain(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
