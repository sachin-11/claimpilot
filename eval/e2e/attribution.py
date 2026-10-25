# ruff: noqa: E501  (long report / explanation strings)
"""Failure attribution: the FIRST layer at which a wrong claim went wrong (pure).

Order: infra -> extraction -> routing -> coverage -> risk -> calculator / decision rules.
A claim is "failed" when its decision is wrong or its payable amount is outside the tolerance.
Coverage failures carry a sub-cause so a data gap (no policy wording ingested) is not mistaken for
a model mistake.
"""

from collections import defaultdict
from decimal import Decimal
from typing import Any

from app.agents.facts import rule_category
from eval.e2e.common import AMOUNT_TOLERANCE

LAYERS = ["infra", "extraction", "routing", "coverage", "risk", "calculator", "decision_rules"]
RULE_CATEGORIES = ("room_rent", "professional_fees", "ot_charges", "icu")


def is_failed(score: dict[str, Any]) -> bool:
    return (not score["decision_correct"]) or score.get("amount_ok") is False


def _close(a: Any, b: Any, tol: float = AMOUNT_TOLERANCE) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(Decimal(str(a))) - float(Decimal(str(b)))) <= tol


def _extracted_totals(extraction: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for item in extraction.get("line_items") or []:
        category = rule_category(item)  # same mapping the decision path applies
        totals[category] += float(Decimal(str(item["amount"])))
    return totals


def extraction_mismatches(extraction: dict[str, Any] | None, truth: dict[str, Any]) -> list[str]:
    """Fields the decision depends on that differ from extraction_truth.json."""
    if not extraction:
        return ["no extraction"]
    fields = truth["fields"]
    bad: list[str] = []
    for key in ("admission_date", "discharge_date"):
        if str(extraction.get(key)) != str(fields.get(key)):
            bad.append(f"{key}: {extraction.get(key)} != {fields.get(key)}")
    if (
        str(extraction.get("icd10_code") or "").upper()
        != str(fields.get("icd10_code") or "").upper()
    ):
        bad.append(f"icd10_code: {extraction.get('icd10_code')} != {fields.get('icd10_code')}")
    for key in ("bill_total", "claimed_amount", "taxes"):
        got = extraction.get(key)
        if key == "taxes" and got is None:
            got = 0  # a bill without tax: the extractor may leave the field empty
        if fields.get(key) is not None and not _close(got, fields[key]):
            bad.append(f"{key}: {extraction.get(key)} != {fields[key]}")
    truth_totals: dict[str, float] = defaultdict(float)
    room_rate = None
    for item in truth.get("line_items") or []:
        truth_totals[item["category"]] += item["amount"]
        if item["category"] == "room_rent" and room_rate is None:
            room_rate = item["rate"]
    got = _extracted_totals(extraction)
    for category in RULE_CATEGORIES:
        if not _close(got.get(category, 0.0), truth_totals.get(category, 0.0)):
            bad.append(
                f"{category} total: {got.get(category, 0.0):.0f} != {truth_totals.get(category, 0.0):.0f}"
            )
    if room_rate is not None and not _close(extraction.get("room_rent_per_day"), room_rate):
        bad.append(f"room_rent_per_day: {extraction.get('room_rent_per_day')} != {room_rate}")
    return bad


def _verdict(layer: str, why: str, sub: str | None = None) -> dict[str, str]:
    return {"layer": layer, "sub_cause": sub or "", "explanation": why}


def attribute(
    score: dict[str, Any],
    golden: dict[str, Any],
    truth: dict[str, Any] | None,
    run: dict[str, Any],
    rule_map: dict[str, Any],
) -> dict[str, str] | None:
    """Layer + one-line explanation for a failed claim, or None when the claim is fine."""
    if not is_failed(score):
        return None
    wording_policies = set(rule_map.get("wording_policies") or [])
    driving_name = driving_check(score, rule_map)
    expected, predicted = score["expected_decision"], score["predicted_decision"]

    # 1. infra
    if not score["ok_run"]:
        errors = run.get("errors") or []
        first = errors[0] if errors else {}
        what = f"{first.get('node', 'run')}: {first.get('error', run.get('status'))}"
        return _verdict("infra", f"run ended '{run.get('status')}' ({what})"[:240])

    # 2. extraction
    if truth is not None:
        bad = extraction_mismatches(run.get("extraction"), truth)
        if bad:
            return _verdict("extraction", "; ".join(bad[:3])[:240])

    # 3. routing
    steps = {s for s in run.get("completed_steps") or [] if ":" not in s}
    if expected == "request_documents" and predicted != "request_documents":
        return _verdict(
            "routing",
            f"missing documents {golden['expected_missing_documents']} not detected (predicted {predicted})",
        )
    if expected != "request_documents" and predicted == "request_documents":
        return _verdict(
            "routing", f"documents wrongly reported missing: {run.get('missing_documents')}"
        )
    if expected != "request_documents" and "coverage" not in steps:
        return _verdict("routing", "coverage step was skipped")

    # 4. coverage
    decision = run.get("decision") or {}
    reasons = " ".join(decision.get("reasons") or []).lower()
    coverage = [c for c in run.get("coverage") or [] if c.get("applies")]
    unclear = [c["check"] for c in coverage if c["result"] == "unclear"]
    if predicted == "flag_for_review" and expected != "flag_for_review":
        if score["policy_id"] not in wording_policies and unclear:
            return _verdict(
                "coverage",
                f"no policy wording ingested for {score['policy_id']}: {', '.join(unclear)} unclear",
                "no_policy_wording",
            )
        if "disagree" in reasons:
            return _verdict(
                "coverage",
                "policy rules and wording disagree on a blocking rule",
                "rules_vs_wording",
            )
        if unclear:
            return _verdict(
                "coverage",
                f"checks without a valid citation: {', '.join(unclear)}",
                "unclear_checks",
            )
    if expected == "reject":
        driving = next((c for c in coverage if c["check"] == driving_name), None)
        if driving is None or driving["result"] != "fail":
            got = driving["result"] if driving else "not evaluated"
            return _verdict(
                "coverage",
                f"driving check {driving_name} is '{got}', expected fail",
                "wrong_result",
            )
    if expected in ("approve", "partial_approve") and any(
        c["check"] in ("initial_waiting", "specific_disease_waiting", "pre_existing", "exclusion")
        and c["result"] == "fail"
        for c in coverage
    ):
        return _verdict("coverage", "a blocking check failed for a payable claim", "wrong_result")

    # 5. risk
    if score["risk_fn"] or (predicted == "flag_for_review" and expected != "flag_for_review"):
        if score["risk_fn"]:
            return _verdict(
                "risk",
                f"expected flags missing: {sorted(set(score['risk_expected']) - set(score['risk_predicted']))}",
            )
        extra = sorted(set(score["risk_predicted"]) - set(score["risk_expected"]))
        return _verdict("risk", f"unexpected risk flags caused a review: {extra}")
    if score["risk_fp"] and predicted == "flag_for_review" and expected != "flag_for_review":
        return _verdict("risk", "false-positive risk flag")

    # 6. calculator / decision rules
    if predicted == expected:
        return _verdict(
            "calculator",
            f"payable {score['payable_pred']} vs expected {score['payable_expected']} (deductions match {score['deduction_match']})",
        )
    return _verdict(
        "decision_rules",
        f"coverage and risk look right but the decision is {predicted}, expected {expected}",
    )


def driving_check(score: dict[str, Any], rule_map: dict[str, Any]) -> str:
    from eval.e2e.metrics import rule_family

    entry = (rule_map.get("rules") or {}).get(rule_family(score["rule_expected"])) or {}
    return entry.get("check", "")


def summarize(verdicts: list[dict[str, str]]) -> dict[str, int]:
    counts = {layer: 0 for layer in LAYERS}
    for v in verdicts:
        counts[v["layer"]] += 1
    return counts
