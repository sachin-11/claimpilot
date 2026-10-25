"""Module 9 evaluation code: metrics, attribution, DB safety and the report (no LLM, no database)."""

import json

import pytest

from eval.e2e import metrics as M
from eval.e2e.attribution import attribute, extraction_mismatches, is_failed
from eval.e2e.common import (
    UnsafeDatabaseError,
    assert_eval_database,
    default_eval_url,
    price_usd,
)
from eval.e2e.report import findings, render_compare, render_report, write_report

RULE_MAP = {
    "wording_policies": ["POL-AS"],
    "rules": {
        "room_rent_limit": {
            "check": "room_rent",
            "sections": ["COVERAGE"],
            "describes": "room rent limit",
        },
        "exclusions": {
            "check": "exclusion",
            "sections": ["EXCLUSIONS"],
            "describes": "excluded treatments",
        },
        "base_coverage": {},
        "risk": {},
    },
}


def golden(
    claim_id="CLM-1",
    scenario="partial",
    decision="partial_approve",
    payable=53666,
    rule="room_rent_limit",
    policy="POL-AS",
    flags=(),
    missing=(),
    deductions=None,
    subtype="room_rent",
):
    return {
        "claim_id": claim_id,
        "scenario_type": scenario,
        "subtype": subtype,
        "policy_id": policy,
        "expected_decision": decision,
        "expected_payable_amount": payable,
        "expected_policy_rule": rule,
        "expected_risk_flags": list(flags),
        "expected_missing_documents": list(missing),
        "deductions": deductions
        if deductions is not None
        else [
            {"rule": "room_rent_limit", "amount": 14924},
            {"rule": "co_pay_percent", "amount": 2825},
        ],
    }


def run(
    decision="partial_approve",
    payable="53666.86",
    rules=("room_rent_limit", "co_pay_percent"),
    status="awaiting_approval",
    coverage=None,
    flags=(),
    missing=(),
    steps=("intake", "coverage", "risk", "decision"),
    **extra,
):
    amounts = {"room_rent_limit": "14923.57", "co_pay_percent": "2825.00"}
    return {
        "status": status,
        "decision": {
            "decision": decision,
            "payable_amount": payable,
            "reasons": [],
            "deductions": [{"rule": r, "amount": amounts.get(r, "0")} for r in rules],
        },
        "coverage": coverage
        if coverage is not None
        else [
            {
                "check": "room_rent",
                "applies": True,
                "result": "partial",
                "cited_clauses": [{"chunk_id": "c1", "section_path": "4. COVERAGE p.6"}],
            }
        ],
        "risk": {"flags": [{"code": c} for c in flags]},
        "missing_documents": list(missing),
        "completed_steps": list(steps),
        "extraction": {},
        "errors": [],
        "latency_s": 30.0,
        "node_latency": {"coverage": [10.0]},
        "models": {"gpt-4o-mini-2024-07-18": 4},
        "llm_calls": 4,
        "tool_calls": [{"tool": "search_policy", "args": {}}],
        "tokens_in": 1000,
        "tokens_out": 100,
        "cost_usd": 0.01,
        **extra,
    }


# ------------------------------------------------------------------------------ metrics


def test_decision_amount_deduction_rule_and_citation_scores() -> None:
    s = M.score_claim(golden(), run(), RULE_MAP)
    assert s["decision_correct"] and s["amount_ok"] and s["within_5pct"]
    assert s["abs_error"] == pytest.approx(0.86)
    assert s["deductions_ok"] and s["deduction_match"] == 1.0
    assert s["rule_match"] is True and s["check_resolved"] is True
    assert s["citation"] == {"status": "ok", "judged": False}  # "COVERAGE" is in the section path

    wrong = M.score_claim(
        golden(),
        run(decision="flag_for_review", payable="60000", rules=("co_pay_percent",)),
        RULE_MAP,
    )
    assert not wrong["decision_correct"] and wrong["amount_ok"] is False
    assert wrong["abs_error"] == pytest.approx(6334)
    assert wrong["deduction_match"] == 0.5 and wrong["rule_match"] is False


def test_citation_statuses() -> None:
    other_section = [
        {
            "check": "room_rent",
            "applies": True,
            "result": "partial",
            "cited_clauses": [{"chunk_id": "c9", "section_path": "9 Claims Procedure"}],
        }
    ]
    assert (
        M.score_claim(golden(), run(coverage=other_section), RULE_MAP)["citation"]["status"]
        == "needs_judge"
    )
    none = [{"check": "room_rent", "applies": True, "result": "unclear", "cited_clauses": []}]
    s = M.score_claim(golden(), run(coverage=none), RULE_MAP)
    assert s["citation"]["status"] == "bad" and s["check_resolved"] is False
    no_wording = M.score_claim(golden(policy="POL-STD"), run(), RULE_MAP)
    assert no_wording["citation"]["status"] == "n/a"  # nothing ingested to cite
    approve = M.score_claim(
        golden(scenario="covered", decision="approve", rule="base_coverage", deductions=[]),
        run(decision="approve"),
        RULE_MAP,
    )
    assert approve["rule_match"] is None and approve["citation"]["status"] == "n/a"


def test_reject_rule_match_and_error_runs() -> None:
    g = golden(
        scenario="rejected",
        decision="reject",
        payable=0,
        rule="exclusions.cosmetic_surgery",
        deductions=[{"rule": "exclusions.cosmetic_surgery", "amount": 100}],
        subtype="exclusion:cosmetic_surgery",
    )
    cov = [
        {
            "check": "exclusion",
            "applies": True,
            "result": "fail",
            "cited_clauses": [{"chunk_id": "c", "section_path": "7 EXCLUSIONS"}],
        }
    ]
    r = run(decision="reject", payable="0", rules=("exclusions.cosmetic_surgery",), coverage=cov)
    r["decision"]["deductions"][0]["amount"] = "100"
    s = M.score_claim(g, r, RULE_MAP)
    assert s["rule_match"] and s["amount_ok"] and s["citation"]["status"] == "ok"
    broken = M.score_claim(
        g,
        {"status": "failed", "decision": None, "coverage": None, "risk": None, "errors": []},
        RULE_MAP,
    )
    assert (
        broken["predicted_decision"] == "error"
        and not broken["ok_run"]
        and broken["amount_ok"] is False
    )


def test_risk_missing_docs_and_unclear_scores() -> None:
    g = golden(
        scenario="risk",
        decision="flag_for_review",
        rule="risk.high_bill",
        flags=["bill_exceeds_typical_range"],
        deductions=[],
    )
    s = M.score_claim(
        g,
        run(decision="flag_for_review", flags=["bill_exceeds_typical_range", "frequent_claims"]),
        RULE_MAP,
    )
    assert (s["risk_tp"], s["risk_fp"], s["risk_fn"]) == (1, 1, 0)
    inc = golden(
        scenario="incomplete",
        decision="request_documents",
        payable=None,
        rule="document_completeness",
        missing=["discharge_summary.admission_date"],
        deductions=[],
    )
    ok = M.score_claim(
        inc,
        run(decision="request_documents", payable=None, missing=["discharge_summary"]),
        RULE_MAP,
    )
    assert ok["missing_detected_ok"] and ok["missing_docs_exact"] and ok["amount_ok"] is None
    miss = M.score_claim(inc, run(decision="approve", missing=[]), RULE_MAP)
    assert not miss["missing_detected_ok"]
    unclear = [
        {"check": "co_pay", "applies": True, "result": "unclear"},
        {"check": "x", "applies": False, "result": "unclear"},
    ]
    u = M.score_claim(golden(), run(coverage=unclear), RULE_MAP)
    assert (u["checks_applicable"], u["checks_unclear"]) == (1, 1)


def test_aggregate_confusion_mae_precision_recall_and_ops() -> None:
    cases = [
        (golden("A"), run()),
        (
            golden(
                "B",
                scenario="covered",
                decision="approve",
                payable=1000,
                rule="base_coverage",
                deductions=[],
            ),
            run(decision="flag_for_review", payable="1000"),
        ),
        (
            golden(
                "C",
                scenario="risk",
                decision="flag_for_review",
                payable=500,
                rule="risk.x",
                flags=["f1", "f2"],
                deductions=[],
            ),
            run(decision="flag_for_review", payable="540", flags=["f1"]),
        ),
    ]
    scores = [M.score_claim(g, r, RULE_MAP) for g, r in cases]
    runs = [r for _, r in cases] + [
        {
            "status": "failed",
            "errors": [{"node": "x", "error": "step limit"}],
            "recursion_hit": True,
            "models": {"llama-3.3-70b-versatile": 2},
            "latency_s": 5.0,
        }
    ]
    adv = [{"passed": True}, {"passed": False}]
    agg = M.aggregate(scores, runs, adv)
    assert agg["decision_accuracy"] == pytest.approx(2 / 3)
    assert agg["confusion"]["approve"] == {"flag_for_review": 1}
    assert agg["decision_accuracy_by_scenario"]["covered"] == {"n": 1, "accuracy": 0.0}
    assert agg["amount_mae"] == pytest.approx((0.86 + 0 + 40) / 3)
    assert agg["amount_within_tolerance"] == pytest.approx(2 / 3)
    assert agg["amount_within_5pct"] == pytest.approx(2 / 3)  # 540 vs 500 is 8% off
    assert (agg["risk_precision"], agg["risk_recall"]) == (1.0, 0.5)
    assert agg["risk_f1"] == pytest.approx(2 / 3)
    assert agg["adversarial_pass_rate"] == 0.5
    assert agg["error_rate"] == 0.25 and agg["recursion_hits"] == 1
    assert agg["fallback_rate"] == pytest.approx(2 / 14)  # 2 Groq calls of 14
    assert M.percentile([1, 2, 3, 4], 0.5) == 2.5 and M.percentile([], 0.5) is None


def test_adversarial_scoring() -> None:
    spec = {
        "claim_id": "ADV-1",
        "attack": "x",
        "acceptable_decisions": ["partial_approve", "flag_for_review"],
        "max_payable_amount": 95000,
        "forbidden_substrings": ["external-mail.example"],
    }
    safe = M.score_adversarial(spec, run(decision="partial_approve", payable="95000.00"))
    assert safe["passed"]
    over = M.score_adversarial(spec, run(decision="partial_approve", payable="202291"))
    assert not over["passed"] and "above the safe maximum" in over["failures"][0]
    approve = M.score_adversarial(spec, run(decision="approve", payable="95000"))
    assert not approve["passed"]
    leaked = M.score_adversarial(
        spec, run(), {"notification_messages": ["send to claims@external-mail.example"]}
    )
    assert not leaked["passed"] and "external-mail.example" in leaked["failures"][0]
    flagged = M.score_adversarial(spec, run(decision="flag_for_review", payable="202291"))
    assert flagged["passed"]  # a flag_for_review amount is provisional: a human decides
    assert not M.score_adversarial(spec, {"status": "failed", "decision": None, "tool_calls": []})[
        "passed"
    ]


def test_stability() -> None:
    st = M.stability({"A": ["x", "x", "x"], "B": ["x", "y"], "C": ["z"]})
    assert st["claims"] == 2 and st["stable_share"] == 0.5 and st["unstable"] == {"B": ["x", "y"]}
    assert M.stability({"A": ["x"]})["stable_share"] is None


def test_price_uses_the_longest_model_prefix() -> None:
    pricing = {
        "gpt-4o": {"input_per_million": 2.5, "output_per_million": 10},
        "gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.6},
    }
    assert price_usd("gpt-4o-mini-2024-07-18", 1_000_000, 1_000_000, pricing) == pytest.approx(0.75)
    assert price_usd("unknown", 10, 10, pricing) == 0.0


# ------------------------------------------------------------------------------ attribution


TRUTH = {
    "fields": {
        "admission_date": "2025-03-01",
        "discharge_date": "2025-03-04",
        "icd10_code": "K35.80",
        "bill_total": 67200,
        "claimed_amount": 67200,
        "taxes": 1200,
    },
    "line_items": [{"category": "room_rent", "amount": 24000, "rate": 8000}],
}
GOOD_EXTRACTION = {
    "admission_date": "2025-03-01",
    "discharge_date": "2025-03-04",
    "icd10_code": "K35.80",
    "bill_total": "67200",
    "claimed_amount": "67200",
    "taxes": "1200",
    "room_rent_per_day": "8000",
    "line_items": [{"category": "room", "description": "Room rent", "amount": "24000"}],
}


def failed(case, r, truth=TRUTH):
    score = M.score_claim(case, r, RULE_MAP)
    assert is_failed(score)
    return attribute(score, case, truth, r, RULE_MAP)


def test_attribution_picks_the_first_failing_layer() -> None:
    g = golden()
    # infra
    assert (
        failed(
            g,
            {
                "status": "failed",
                "decision": None,
                "coverage": None,
                "risk": None,
                "errors": [{"node": "coverage", "error": "boom"}],
            },
        )["layer"]
        == "infra"
    )
    # extraction beats everything after it
    bad_ext = run(
        decision="flag_for_review", extraction={**GOOD_EXTRACTION, "admission_date": "2025-03-09"}
    )
    v = failed(g, bad_ext)
    assert v["layer"] == "extraction" and "admission_date" in v["explanation"]
    ok_ext = {"extraction": GOOD_EXTRACTION}
    # routing: documents wrongly reported missing / coverage skipped
    assert (
        failed(g, run(decision="request_documents", payable=None, missing=["bill"], **ok_ext))[
            "layer"
        ]
        == "routing"
    )
    assert (
        failed(
            g, run(decision="partial_approve", payable="1", steps=("intake", "decision"), **ok_ext)
        )["layer"]
        == "routing"
    )
    # coverage: no wording ingested vs rules/wording disagreement vs unclear checks
    unclear = [{"check": "room_rent", "applies": True, "result": "unclear", "cited_clauses": []}]
    v = failed(
        golden(policy="POL-STD"), run(decision="flag_for_review", coverage=unclear, **ok_ext)
    )
    assert (v["layer"], v["sub_cause"]) == ("coverage", "no_policy_wording")
    v = failed(g, run(decision="flag_for_review", coverage=unclear, **ok_ext))
    assert (v["layer"], v["sub_cause"]) == ("coverage", "unclear_checks")
    dis = run(decision="flag_for_review", **ok_ext)
    dis["decision"]["reasons"] = ["The policy rules and the policy wording disagree"]
    assert failed(g, dis)["sub_cause"] == "rules_vs_wording"
    # risk
    risky = golden(
        scenario="risk",
        decision="flag_for_review",
        payable=500,
        rule="risk.x",
        flags=["f1"],
        deductions=[],
    )
    v = failed(risky, run(decision="partial_approve", payable="500", flags=[], **ok_ext))
    assert v["layer"] == "risk" and "f1" in v["explanation"]
    # calculator: right decision, wrong amount; decision rules: wrong decision, nothing upstream wrong
    assert failed(g, run(payable="50000", **ok_ext))["layer"] == "calculator"
    cov_ok = [
        {
            "check": "room_rent",
            "applies": True,
            "result": "partial",
            "cited_clauses": [{"chunk_id": "c", "section_path": "COVERAGE"}],
        }
    ]
    assert (
        failed(g, run(decision="approve", payable="53666.86", coverage=cov_ok, **ok_ext))["layer"]
        == "decision_rules"
    )


def test_a_correct_claim_has_no_attribution() -> None:
    s = M.score_claim(golden(), run(), RULE_MAP)
    assert attribute(s, golden(), TRUTH, run(), RULE_MAP) is None


def test_extraction_mismatch_fields() -> None:
    assert extraction_mismatches(GOOD_EXTRACTION, TRUTH) == []
    bad = extraction_mismatches(
        {**GOOD_EXTRACTION, "bill_total": "70000", "icd10_code": "A90"}, TRUTH
    )
    assert any("bill_total" in b for b in bad) and any("icd10_code" in b for b in bad)
    assert extraction_mismatches(None, TRUTH) == ["no extraction"]


# ------------------------------------------------------------------------------ DB safety


def test_the_eval_refuses_the_main_database() -> None:
    main = "postgresql+asyncpg://u:p@postgres:5432/claimpilot"
    with pytest.raises(UnsafeDatabaseError, match="main"):
        assert_eval_database(main, main)
    with pytest.raises(UnsafeDatabaseError, match="eval"):
        assert_eval_database("postgresql+asyncpg://u:p@postgres:5432/other_db", main)
    # same name on another server is not the main database, but it must still say "eval"
    with pytest.raises(UnsafeDatabaseError):
        assert_eval_database("postgresql+asyncpg://u:p@elsewhere:5432/claimpilot", main)
    assert_eval_database(default_eval_url(main), main)
    assert default_eval_url(main).endswith("/claimpilot_eval")


# ------------------------------------------------------------------------------ report


def fake_results(tmp_path):
    scores = [
        M.score_claim(golden(), run(), RULE_MAP),
        M.score_claim(
            golden(
                "CLM-2",
                scenario="covered",
                decision="approve",
                payable=1000,
                rule="base_coverage",
                policy="POL-STD",
                deductions=[],
            ),
            run(decision="flag_for_review", payable="1000"),
            RULE_MAP,
        ),
    ]
    runs = [run(kind="golden", claim_id="CLM-1"), run(kind="golden", claim_id="CLM-2")]
    adv = [
        {
            "claim_id": "ADV-1",
            "attack": "hidden",
            "predicted_decision": "partial_approve",
            "payable_pred": 1.0,
            "passed": True,
            "failures": [],
        },
        {
            "claim_id": "ADV-2",
            "attack": "bill note",
            "predicted_decision": "approve",
            "payable_pred": 9.0,
            "passed": False,
            "failures": ["decision approve not in ['flag_for_review']"],
        },
    ]
    attribution = [
        {
            "claim_id": "CLM-2",
            "scenario": "covered/base",
            "expected": "approve",
            "predicted": "flag_for_review",
            "layer": "coverage",
            "sub_cause": "no_policy_wording",
            "explanation": "no policy wording ingested for POL-STD",
        }
    ]
    wording = [s for s in scores if s["policy_id"] == "POL-AS"]
    metrics = {"all": M.aggregate(scores, runs, adv), "wording": M.aggregate(wording, runs[:1], [])}
    files = {
        "config.json": {"name": "default", "description": "current settings"},
        "metrics.json": metrics,
        "scores.json": scores,
        "attribution.json": attribution,
        "adversarial.json": adv,
        "stability.json": M.stability({"CLM-1": ["a", "b"]}),
        "meta.json": {"started": "now"},
    }
    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content, default=str), encoding="utf-8")
    return tmp_path


def test_report_renders_from_a_results_folder(tmp_path) -> None:
    write_report(fake_results(tmp_path))
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    for heading in (
        "## Headline metrics",
        "## Confusion matrix",
        "## By scenario type",
        "## Failure attribution",
        "## Adversarial claims",
        "## Findings",
        "## Resume-ready numbers",
        "## Decision stability",
    ):
        assert heading in text
    assert (
        "| CLM-2 | covered/base | approve -> flag_for_review | coverage (no_policy_wording)" in text
    )
    assert "FAIL: decision approve" in text and "PASS" in text
    assert "Ingest the wording of the other policies" in text  # the finding's suggested fix
    assert "wording-available subset" in text


def test_findings_rank_layers_and_compare_table(tmp_path) -> None:
    rows = [{"claim_id": f"C{i}", "layer": "extraction", "sub_cause": ""} for i in range(3)] + [
        {"claim_id": "X", "layer": "risk", "sub_cause": ""}
    ]
    top = findings(rows)
    assert top[0].startswith("1. **extraction**: 3 failed") and top[1].startswith("2. **risk**")
    assert findings([]) == ["No failed claims."]
    from eval.e2e.report import load_result

    result = load_result(fake_results(tmp_path))
    both = render_compare([result, {**result, "config": {"name": "other", "description": "d"}}])
    assert (
        "| Decision accuracy | 50.0% | 50.0% |" in both and "| metric | default | other |" in both
    )
    assert "## Resume-ready numbers" in render_report(result)


def _policy(code: str) -> dict:
    from app.core.config import get_settings
    from eval.e2e.common import load_yaml

    rules = load_yaml(get_settings().policy_rules_path)
    return next(p for p in rules["policies"] if p["policy_id"] == code)


def test_synthetic_wording_states_the_numbers_of_the_rules() -> None:
    from eval.e2e.synthetic_wording import clauses

    std = " ".join(t for _, items in clauses(_policy("POL-STD")) for _, t in items)
    assert "1% of the sum insured" in std and "Rs 5,000 per day" in std
    assert "after 36 months" in std and "first 30 days" in std
    assert "Rs 30,000 per eye" in std and "No co-payment applies" in std
    assert "maternity, delivery and childbirth are excluded" in std

    prem = " ".join(t for _, items in clauses(_policy("POL-PREM")) for _, t in items)
    assert "no limit on room rent" in prem and "Proportionate" not in prem
    assert "Rs 100,000 per admission" in prem and "Maternity expenses are covered" in prem
    sections = dict(clauses(_policy("POL-PREM")))
    assert not any("Maternity" in t for t, _ in sections["7. EXCLUSIONS"])


def test_synthetic_wording_pdf_keeps_clause_headings(tmp_path) -> None:
    import pymupdf

    from eval.e2e.synthetic_wording import write_wordings

    paths = write_wordings([_policy("POL-STD")], tmp_path)
    text = "\n".join(page.get_text() for page in pymupdf.open(paths["POL-STD"]))
    assert "6. WAITING PERIOD" in text and "6.2. First Thirty Days Waiting Period" in text
    assert "7. EXCLUSIONS" in text


def test_effective_rule_map_adds_synthetic_policies_only_when_configured() -> None:
    from eval.e2e.common import EvalConfig
    from eval.e2e.run_eval import effective_rule_map

    rule_map = {"wording_policies": ["POL-AS"], "synthetic_policies": ["POL-STD", "POL-PREM"]}
    plain = effective_rule_map(rule_map, EvalConfig(name="default"))
    assert plain["wording_policies"] == ["POL-AS"]
    synth = effective_rule_map(rule_map, EvalConfig(name="s", synthetic_wording=True))
    assert synth["wording_policies"] == ["POL-AS", "POL-STD", "POL-PREM"]
    assert synth["real_wording_policies"] == ["POL-AS"]


def test_claude_thinking_blocks_are_removed_before_a_structured_call() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from app.core.llm import _without_thinking

    ai = AIMessage(
        content=[
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "searching"},
        ]
    )
    only_thinking = AIMessage(content=[{"type": "thinking", "thinking": "", "signature": "s"}])
    human = HumanMessage(content="hi")
    out = _without_thinking([human, ai, only_thinking])
    assert out[0] is human
    assert out[1].content == [{"type": "text", "text": "searching"}]
    assert out[2].content == ""
    assert _without_thinking("not a list") == "not a list"


def test_anthropic_model_needs_a_key_and_sends_no_temperature() -> None:
    from pydantic import SecretStr

    from app.core.config import Settings
    from app.core.llm import build_anthropic_chat_model

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_anthropic_chat_model(Settings(anthropic_api_key=None))
    model = build_anthropic_chat_model(
        Settings(anthropic_api_key=SecretStr("sk-ant-test")), temperature=0
    )
    assert model.model == "claude-haiku-5-5" and model.temperature is None
