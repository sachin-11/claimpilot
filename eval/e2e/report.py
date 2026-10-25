# ruff: noqa: E501  (long report / explanation strings)
"""Render report.md from a results folder (pure: reads the JSON files the runner wrote)."""

import json
from pathlib import Path
from typing import Any

from eval.e2e.attribution import LAYERS
from eval.e2e.metrics import DECISIONS

SUGGESTIONS = {
    ("coverage", "no_policy_wording"): (
        "Ingest the wording of the other policies (only POL-AS has a PDF). Until then claims on those "
        "policies cannot be confirmed with a citation and correctly end in flag_for_review."
    ),
    ("coverage", "rules_vs_wording"): (
        "Reconcile policies.rules with the policy PDF (they differ, e.g. pre-existing waiting 48 vs 36 "
        "months), or make the wording the single source for those numbers."
    ),
    ("coverage", "unclear_checks"): (
        "Improve retrieval for the checks that end without a citation: better search queries per check "
        "(relevance.py), a reranker, or a larger tool-call budget."
    ),
    ("coverage", "wrong_result"): (
        "Tighten the coverage prompt for the failing check and add a deterministic cross-check like the "
        "calculator reconciliation already used for amount checks."
    ),
    ("extraction", ""): (
        "Fix the extraction on the failing fields (prompt, OCR settings, or a deterministic parser for "
        "the structured claim form); extraction errors change every number downstream."
    ),
    (
        "routing",
        "",
    ): "Check missing-document detection in intake (Module 6) and the supervisor guards.",
    ("risk", ""): "Tune the risk rules/thresholds; compare the flag lists in the table below.",
    ("calculator", ""): "Compare the calculator with the golden calculation for the listed claims.",
    ("decision_rules", ""): "Review the decision rule priority for the listed claims.",
    (
        "infra",
        "",
    ): "Look at the errors (timeouts, provider failures, recursion limit) in runs.jsonl.",
}


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 1, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def load_result(folder: Path) -> dict[str, Any]:
    def read(name: str, default: Any) -> Any:
        path = folder / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

    return {
        "config": read("config.json", {}),
        "metrics": read("metrics.json", {}),
        "scores": read("scores.json", []),
        "attribution": read("attribution.json", []),
        "adversarial": read("adversarial.json", []),
        "stability": read("stability.json", {}),
        "meta": read("meta.json", {}),
    }


def headline_rows(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("Decision accuracy", pct(metrics.get("decision_accuracy"))),
        ("Amount within Rs 1", pct(metrics.get("amount_within_tolerance"))),
        ("Amount within 5%", pct(metrics.get("amount_within_5pct"))),
        ("Amount MAE (Rs)", num(metrics.get("amount_mae"), 0)),
        ("Deductions match", pct(metrics.get("deductions_match"))),
        ("Rule match (driving check)", pct(metrics.get("rule_match"))),
        ("Driving check resolved (not unclear)", pct(metrics.get("check_resolved"))),
        (
            "Citation correctness",
            f"{pct(metrics.get('citation_correctness'))} (n={metrics.get('citation_n', 0)}, "
            f"judged {pct(metrics.get('citation_judged_share'), 0)})",
        ),
        ("Unclear rate (applicable checks)", pct(metrics.get("unclear_rate"))),
        (
            "Risk flag precision / recall / F1",
            f"{pct(metrics.get('risk_precision'))} / {pct(metrics.get('risk_recall'))} / {pct(metrics.get('risk_f1'))}",
        ),
        ("Missing-docs detection", pct(metrics.get("missing_docs_accuracy"))),
        (
            "Adversarial pass rate",
            f"{pct(metrics.get('adversarial_pass_rate'))} (n={metrics.get('adversarial_n', 0)})",
        ),
        ("Error rate", pct(metrics.get("error_rate"))),
        (
            "Latency p50 / p95 (s)",
            f"{num(metrics.get('latency_p50_s'))} / {num(metrics.get('latency_p95_s'))}",
        ),
        (
            "LLM calls / tool calls per claim",
            f"{num(metrics.get('llm_calls_per_claim'))} / {num(metrics.get('tool_calls_per_claim'))}",
        ),
        ("Tokens per claim", num(metrics.get("tokens_per_claim"), 0)),
        ("Cost per claim (USD, estimate)", num(metrics.get("cost_per_claim_usd"), 4)),
        ("Fallback (Groq / other) share of LLM calls", pct(metrics.get("fallback_rate"))),
        ("Recursion-limit hits", str(metrics.get("recursion_hits", 0))),
    ]


def confusion_table(confusion: dict[str, dict[str, int]]) -> str:
    cols = [*DECISIONS, "error"]
    lines = ["| expected \\ predicted | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for expected in DECISIONS:
        row = confusion.get(expected)
        if not row:
            continue
        lines.append(
            f"| **{expected}** | " + " | ".join(str(row.get(c, 0)) or "0" for c in cols) + " |"
        )
    return "\n".join(lines)


def findings(attribution: list[dict[str, Any]]) -> list[str]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for a in attribution:
        groups.setdefault((a["layer"], a.get("sub_cause", "")), []).append(a)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:3]
    out = []
    for i, ((layer, sub), items) in enumerate(ranked, 1):
        ids = ", ".join(a["claim_id"] for a in items[:8]) + (" ..." if len(items) > 8 else "")
        suggestion = SUGGESTIONS.get((layer, sub)) or SUGGESTIONS.get((layer, ""), "")
        label = f"{layer}" + (f" / {sub}" if sub else "")
        out.append(
            f"{i}. **{label}**: {len(items)} failed claim(s) ({ids}). Suggested fix: {suggestion}"
        )
    return out or ["No failed claims."]


def resume_lines(m_all: dict[str, Any], m_wording: dict[str, Any], n_adv: int) -> list[str]:
    lines = [
        f"- End-to-end agent decision accuracy of {pct(m_all.get('decision_accuracy'), 0)} on "
        f"{m_all.get('claims', 0)} golden claims ({pct(m_wording.get('decision_accuracy'), 0)} on the "
        f"{m_wording.get('claims', 0)} claims whose policy wording is ingested).",
        f"- Payable amount within Rs 1 of the golden amount on {pct(m_all.get('amount_within_tolerance'), 0)} of claims.",
        f"- {pct(m_all.get('adversarial_pass_rate'), 0)} of {n_adv} prompt-injection / policy-evasion claims handled safely.",
        f"- Risk flags: precision {pct(m_all.get('risk_precision'), 0)}, recall {pct(m_all.get('risk_recall'), 0)}.",
        f"- p95 latency {num(m_all.get('latency_p95_s'), 0)} s and about ${num(m_all.get('cost_per_claim_usd'), 3)} per claim "
        f"({num(m_all.get('llm_calls_per_claim'), 0)} LLM calls, {num(m_all.get('tool_calls_per_claim'), 0)} MCP tool calls).",
    ]
    if m_wording.get("citation_correctness") is not None:
        lines.append(
            f"- Citation correctness {pct(m_wording.get('citation_correctness'), 0)} "
            f"(n={m_wording.get('citation_n', 0)}, wording-available claims)."
        )
    return lines


def render_report(result: dict[str, Any]) -> str:
    cfg, metrics = result["config"], result["metrics"]
    m_all, m_word = metrics.get("all", {}), metrics.get("wording", {})
    meta = result.get("meta", {})
    out = [f"# End-to-end evaluation: `{cfg.get('name', '?')}`", ""]
    out.append(f"{cfg.get('description', '')}")
    out.append("")
    out.append(
        f"Claims: {m_all.get('claims', 0)} golden + {m_all.get('adversarial_n', 0)} adversarial. "
        f"Evaluated at the human-approval interrupt (nothing approved). Run started {meta.get('started', '?')}."
    )
    out += [
        "",
        "## Headline metrics",
        "",
        "| metric | all golden claims | wording-available subset |",
        "|---|---|---|",
    ]
    for (name, a), (_, w) in zip(
        headline_rows(m_all), headline_rows(m_word) or headline_rows({}), strict=False
    ):
        out.append(f"| {name} | {a} | {w} |")
    out += [
        "",
        "The *wording-available subset* is the claims on policies whose REAL PDF is ingested (POL-AS). "
        + (
            "In this config POL-STD and POL-PREM also have a SYNTHETIC wording generated from "
            "policy_rules.yaml (wording and rules agree by design), so the all-claims column "
            "measures the pipeline, not reading of real wording; only the subset column is real."
            if cfg.get("synthetic_wording")
            else "For the other policies the agent has no clause to cite, so it correctly refuses "
            "to auto-decide."
        ),
        "",
        "## Confusion matrix (all golden claims)",
        "",
        confusion_table(m_all.get("confusion", {})),
        "",
        "## By scenario type",
        "",
        "| scenario | claims | decision accuracy |",
        "|---|---|---|",
    ]
    for scenario, v in (m_all.get("decision_accuracy_by_scenario") or {}).items():
        out.append(f"| {scenario} | {v['n']} | {pct(v['accuracy'])} |")

    attribution = result["attribution"]
    counts = {layer: 0 for layer in LAYERS}
    for a in attribution:
        counts[a["layer"]] += 1
    out += ["", "## Failure attribution", "", "| layer | failed claims |", "|---|---|"]
    out += [f"| {layer} | {counts[layer]} |" for layer in LAYERS]
    if attribution:
        out += [
            "",
            "| claim | scenario | expected -> predicted | layer | why |",
            "|---|---|---|---|---|",
        ]
        for a in attribution:
            layer = a["layer"] + (f" ({a['sub_cause']})" if a.get("sub_cause") else "")
            why = a["explanation"].replace("|", "/")
            out.append(
                f"| {a['claim_id']} | {a['scenario']} | {a['expected']} -> {a['predicted']} | {layer} | {why} |"
            )

    adv = result["adversarial"]
    out += [
        "",
        "## Adversarial claims",
        "",
        "| claim | attack | decision | payable | result |",
        "|---|---|---|---|---|",
    ]
    for a in adv:
        verdict = "PASS" if a["passed"] else "FAIL: " + "; ".join(a["failures"])
        out.append(
            f"| {a['claim_id']} | {a['attack']} | {a['predicted_decision']} | {a['payable_pred']} | {verdict} |"
        )

    out += ["", "## Latency per node (seconds)", "", "| node | p50 | p95 |", "|---|---|---|"]
    p95 = m_all.get("node_latency_p95_s", {})
    for node, p in (m_all.get("node_latency_p50_s") or {}).items():
        out.append(f"| {node} | {num(p, 2)} | {num(p95.get(node), 2)} |")

    stab = result.get("stability") or {}
    if stab.get("claims"):
        out += [
            "",
            "## Decision stability across repeats",
            "",
            f"{pct(stab['stable_share'])} of {stab['claims']} claims kept the same decision in every repeat.",
        ]
        for claim, decisions in (stab.get("unstable") or {}).items():
            out.append(f"- {claim}: {decisions}")

    out += ["", "## Findings", ""] + findings(attribution)
    out += ["", "## Resume-ready numbers", ""] + resume_lines(m_all, m_word, len(adv))
    return "\n".join(out) + "\n"


def render_compare(results: list[dict[str, Any]]) -> str:
    """One table across configs (headline metrics of the all-claims column)."""
    names = [r["config"].get("name", "?") for r in results]
    rows = [headline_rows(r["metrics"].get("all", {})) for r in results]
    out = [
        "# Configuration comparison",
        "",
        "| metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for i, (label, _) in enumerate(rows[0]):
        out.append(f"| {label} | " + " | ".join(r[i][1] for r in rows) + " |")
    out += ["", "Descriptions:"] + [
        f"- **{r['config'].get('name')}**: {r['config'].get('description', '')}" for r in results
    ]
    return "\n".join(out) + "\n"


def write_report(folder: Path) -> Path:
    path = folder / "report.md"
    path.write_text(render_report(load_result(folder)), encoding="utf-8")
    return path
