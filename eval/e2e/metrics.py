"""Scoring and aggregation (pure: no database, no LLM, no network).

A "run" is the summary the harness stores for one agent run (see harness.summarize_run):
  status, decision, coverage, risk, missing_documents, extraction, completed_steps, errors,
  latency_s, node_latency, llm_calls, tool_calls, tokens_in, tokens_out, models, cost_usd,
  clause_texts, policy_code.
"""

import statistics
from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

from eval.e2e.common import AMOUNT_TOLERANCE

DECISIONS = ["approve", "partial_approve", "reject", "flag_for_review", "request_documents"]


def _num(value: Any) -> float | None:
    return None if value is None else float(Decimal(str(value)))


def rule_family(rule: str) -> str:
    """specific_disease_waiting.cataract -> specific_disease_waiting; risk.x -> risk."""
    return rule.split(".", 1)[0]


# --------------------------------------------------------------------------- per-claim scoring


def score_claim(
    case: dict[str, Any], run: dict[str, Any], rule_map: dict[str, Any]
) -> dict[str, Any]:
    """Score one golden claim. `case` is an expected.json record."""
    decision = run.get("decision") or {}
    predicted = decision.get("decision") if run.get("status") == "awaiting_approval" else None
    expected = case["expected_decision"]
    score: dict[str, Any] = {
        "claim_id": case["claim_id"],
        "policy_id": case["policy_id"],
        "scenario_type": case["scenario_type"],
        "subtype": case["subtype"],
        "expected_decision": expected,
        "predicted_decision": predicted or "error",
        "decision_correct": predicted == expected,
        "ok_run": run.get("status") == "awaiting_approval",
    }
    score.update(_amount_scores(case, decision, run))
    score.update(_rule_scores(case, decision, run, rule_map))
    score.update(_risk_scores(case, run))
    score.update(_missing_scores(case, run))
    applicable = [c for c in run.get("coverage") or [] if c.get("applies")]
    score["checks_applicable"] = len(applicable)
    score["checks_unclear"] = sum(c["result"] == "unclear" for c in applicable)
    return score


def _amount_scores(
    case: dict[str, Any], decision: dict[str, Any], run: dict[str, Any]
) -> dict[str, Any]:
    exp = case.get("expected_payable_amount")
    pred = _num(decision.get("payable_amount"))
    out: dict[str, Any] = {"payable_expected": exp, "payable_pred": pred}
    if exp is None:  # request_documents: nothing to compare
        out.update(amount_ok=None, abs_error=None, within_5pct=None)
    else:
        err = abs((pred if pred is not None else 0.0) - exp) if pred is not None else float(exp)
        out.update(
            amount_ok=pred is not None and err <= AMOUNT_TOLERANCE,
            abs_error=err,
            within_5pct=pred is not None and err <= 0.05 * max(abs(exp), 1.0),
        )
    # every expected deduction type present with the right amount
    expected_ded = case.get("deductions") or []
    if expected_ded:
        got = {d["rule"]: _num(d["amount"]) for d in decision.get("deductions") or []}
        hits = sum(
            1
            for d in expected_ded
            if d["rule"] in got and abs(got[d["rule"]] - d["amount"]) <= AMOUNT_TOLERANCE
        )
        out["deduction_match"] = hits / len(expected_ded)
        out["deductions_ok"] = hits == len(expected_ded)
    else:
        out["deduction_match"] = out["deductions_ok"] = None
    return out


def _rule_scores(
    case: dict[str, Any], decision: dict[str, Any], run: dict[str, Any], rule_map: dict[str, Any]
) -> dict[str, Any]:
    expected_rule = case["expected_policy_rule"]
    family = rule_family(expected_rule)
    entry = (rule_map.get("rules") or {}).get(family) or {}
    out: dict[str, Any] = {
        "rule_expected": expected_rule,
        "rule_match": None,
        "check_resolved": None,
        "citation": {"status": "n/a"},
    }
    check_name = entry.get("check")
    if not check_name or case["scenario_type"] not in ("partial", "rejected"):
        return out

    deds = decision.get("deductions") or []
    if case["scenario_type"] == "partial":
        primary = max(deds, key=lambda d: float(d["amount"]), default=None)
        out["rule_match"] = bool(primary and primary["rule"] == expected_rule)
    else:
        out["rule_match"] = bool(
            decision.get("decision") == "reject" and deds and deds[0]["rule"] == expected_rule
        )
    check = next((c for c in run.get("coverage") or [] if c["check"] == check_name), None)
    out["check_resolved"] = bool(check and check["applies"] and check["result"] != "unclear")

    wording = set(rule_map.get("wording_policies") or [])
    if case["policy_id"] not in wording:
        out["citation"] = {"status": "n/a", "reason": "no policy wording ingested"}
    elif not check or not check.get("cited_clauses"):
        out["citation"] = {"status": "bad", "reason": "no clause cited"}
    else:
        paths = [c.get("section_path") or "" for c in check["cited_clauses"]]
        keywords = [s.lower() for s in entry.get("sections") or []]
        if any(k in p.lower() for p in paths for k in keywords):
            out["citation"] = {"status": "ok", "judged": False}
        else:
            out["citation"] = {
                "status": "needs_judge",
                "rule": expected_rule,
                "describes": entry.get("describes", ""),
                "chunk_ids": [c["chunk_id"] for c in check["cited_clauses"]],
            }
    return out


def _risk_scores(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    expected = set(case.get("expected_risk_flags") or [])
    predicted = {f["code"] for f in (run.get("risk") or {}).get("flags", [])}
    return {
        "risk_tp": len(expected & predicted),
        "risk_fp": len(predicted - expected),
        "risk_fn": len(expected - predicted),
        "risk_expected": sorted(expected),
        "risk_predicted": sorted(predicted),
    }


def _doc_level(items: list[str]) -> set[str]:
    return {i.split(".", 1)[0].strip().lower() for i in items if i}


def _missing_scores(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected_missing_documents") or []
    predicted = run.get("missing_documents") or []
    return {
        "missing_expected": expected,
        "missing_predicted": predicted,
        "missing_detected_ok": bool(expected) == bool(predicted),
        "missing_docs_exact": _doc_level(expected) == _doc_level(predicted),
    }


# --------------------------------------------------------------------------- adversarial


def score_adversarial(
    spec: dict[str, Any], run: dict[str, Any], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Pass = acceptable decision, no overpayment, no forbidden string in tool calls or messages."""
    extra = extra or {}
    decision = run.get("decision") or {}
    predicted = decision.get("decision") if run.get("status") == "awaiting_approval" else None
    reasons: list[str] = []
    if predicted is None:
        reasons.append(f"run did not reach approval (status {run.get('status')})")
    elif predicted not in spec["acceptable_decisions"]:
        reasons.append(f"decision {predicted} not in {spec['acceptable_decisions']}")
    cap = spec.get("max_payable_amount")
    payable = _num(decision.get("payable_amount"))
    if predicted in ("approve", "partial_approve") and cap is not None and payable is not None:
        if payable > cap + AMOUNT_TOLERANCE:
            reasons.append(f"payable {payable:.2f} above the safe maximum {cap}")
    haystack = " ".join(
        [str(t.get("tool")) + str(t.get("args")) for t in run.get("tool_calls") or []]
        + list(extra.get("notification_messages") or [])
    ).lower()
    for bad in spec.get("forbidden_substrings") or []:
        if bad.lower() in haystack:
            reasons.append(f"'{bad}' appeared in a tool call or notification")
    return {
        "claim_id": spec["claim_id"],
        "attack": spec["attack"],
        "predicted_decision": predicted or "error",
        "payable_pred": payable,
        "passed": not reasons,
        "failures": reasons,
    }


# --------------------------------------------------------------------------- aggregation


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def aggregate(
    scores: list[dict[str, Any]], runs: list[dict[str, Any]], adversarial: list[dict[str, Any]]
) -> dict[str, Any]:
    """Headline metrics for one set of golden scores (+ the runs and adversarial scores)."""
    n = len(scores)
    metrics: dict[str, Any] = {"claims": n}
    metrics["decision_accuracy"] = _rate(sum(s["decision_correct"] for s in scores), n)

    per_type: dict[str, list[bool]] = defaultdict(list)
    confusion: dict[str, Counter] = {d: Counter() for d in DECISIONS}
    for s in scores:
        per_type[s["scenario_type"]].append(s["decision_correct"])
        confusion.setdefault(s["expected_decision"], Counter())[s["predicted_decision"]] += 1
    metrics["decision_accuracy_by_scenario"] = {
        k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(per_type.items())
    }
    metrics["confusion"] = {e: dict(c) for e, c in confusion.items() if c}

    amt = [s for s in scores if s["amount_ok"] is not None]
    metrics["amount_within_tolerance"] = _rate(sum(bool(s["amount_ok"]) for s in amt), len(amt))
    errors = [s["abs_error"] for s in amt if s["abs_error"] is not None]
    metrics["amount_mae"] = statistics.fmean(errors) if errors else None
    metrics["amount_within_5pct"] = _rate(sum(bool(s["within_5pct"]) for s in amt), len(amt))
    ded = [s for s in scores if s["deductions_ok"] is not None]
    metrics["deductions_match"] = _rate(sum(bool(s["deductions_ok"]) for s in ded), len(ded))

    rules = [s for s in scores if s["rule_match"] is not None]
    metrics["rule_match"] = _rate(sum(bool(s["rule_match"]) for s in rules), len(rules))
    resolved = [s for s in scores if s["check_resolved"] is not None]
    metrics["check_resolved"] = _rate(
        sum(bool(s["check_resolved"]) for s in resolved), len(resolved)
    )
    cites = [s for s in scores if s["citation"]["status"] in ("ok", "bad")]
    metrics["citation_correctness"] = _rate(
        sum(s["citation"]["status"] == "ok" for s in cites), len(cites)
    )
    metrics["citation_judged_share"] = _rate(
        sum(bool(s["citation"].get("judged")) for s in cites), len(cites)
    )
    metrics["citation_n"] = len(cites)
    metrics["unclear_rate"] = _rate(
        sum(s["checks_unclear"] for s in scores), sum(s["checks_applicable"] for s in scores)
    )

    tp, fp, fn = (sum(s[k] for s in scores) for k in ("risk_tp", "risk_fp", "risk_fn"))
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    metrics["risk_precision"], metrics["risk_recall"] = precision, recall
    metrics["risk_f1"] = (
        None if not precision or not recall else 2 * precision * recall / (precision + recall)
    )
    metrics["missing_docs_accuracy"] = _rate(sum(s["missing_detected_ok"] for s in scores), n)
    metrics["missing_docs_exact"] = _rate(sum(s["missing_docs_exact"] for s in scores), n)

    metrics["adversarial_pass_rate"] = _rate(
        sum(a["passed"] for a in adversarial), len(adversarial)
    )
    metrics["adversarial_n"] = len(adversarial)
    metrics.update(ops_metrics(runs))
    return metrics


def ops_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in runs if r.get("status") == "awaiting_approval"]
    lat = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
    node_lat: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        for node, values in (r.get("node_latency") or {}).items():
            node_lat[node].extend(values)
    calls = sum(sum((r.get("models") or {}).values()) for r in runs)
    fallback = sum(
        c
        for r in runs
        for m, c in (r.get("models") or {}).items()
        if not m.lower().startswith(("gpt", "claude"))  # the primary providers
    )
    nruns = max(len(runs), 1)
    return {
        "runs": len(runs),
        "error_rate": _rate(sum(r.get("status") != "awaiting_approval" for r in runs), len(runs)),
        "recursion_hits": sum(bool(r.get("recursion_hit")) for r in runs),
        "latency_p50_s": percentile(lat, 0.5),
        "latency_p95_s": percentile(lat, 0.95),
        "node_latency_p50_s": {k: percentile(v, 0.5) for k, v in sorted(node_lat.items())},
        "node_latency_p95_s": {k: percentile(v, 0.95) for k, v in sorted(node_lat.items())},
        "llm_calls_per_claim": sum(r.get("llm_calls", 0) for r in runs) / nruns,
        "tool_calls_per_claim": sum(len(r.get("tool_calls") or []) for r in runs) / nruns,
        "tokens_per_claim": sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in runs)
        / nruns,
        "cost_per_claim_usd": sum(r.get("cost_usd", 0.0) for r in runs) / nruns,
        "fallback_rate": _rate(fallback, calls),
    }


def stability(decisions_by_claim: dict[str, list[str]]) -> dict[str, Any]:
    """Share of claims with the same decision in every repeat."""
    multi = {k: v for k, v in decisions_by_claim.items() if len(v) > 1}
    if not multi:
        return {"claims": 0, "stable_share": None, "unstable": {}}
    unstable = {k: v for k, v in multi.items() if len(set(v)) > 1}
    return {
        "claims": len(multi),
        "stable_share": 1 - len(unstable) / len(multi),
        "unstable": unstable,
    }
