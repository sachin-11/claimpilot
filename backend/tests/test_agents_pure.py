"""Pure logic of the agents (Module 8): calculator, decision rules, risk rules, outcome mapping,
supervisor guards, citation validation. No DB, no LLM."""

import json
from datetime import date
from decimal import Decimal

import pytest

from app.agents.calculator import (
    ClaimFacts,
    PolicyRules,
    check_eligibility,
    compute_payout,
)
from app.agents.catalog import resolve_diagnosis
from app.agents.config import AgentSettings
from app.agents.decision_rules import decide
from app.agents.facts import build_facts
from app.agents.nodes.coverage import (
    CoverageCheckLLM,
    CoverageLLMOutput,
    known_clauses,
    validate_checks,
)
from app.agents.nodes.decision import reasons_are_grounded
from app.agents.nodes.risk import ExtraFlagLLM, merge_extra_flags
from app.agents.outcome import ApprovalError, final_outcome
from app.agents.risk_rules import risk_score, rule_flags
from app.agents.state import CoverageCheck, HumanApproval, RiskAssessment, RiskFlag
from app.agents.supervisor import SupervisorChoice, allowed_steps, resolve_choice
from app.models import ClaimStatus

D = Decimal
AS = PolicyRules.model_validate(
    {
        "room_rent_limit": {"percent_of_si_per_day": 2.0, "max_per_day": 5000},
        "icu_limit": {"percent_of_si_per_day": 5.0, "max_per_day": 10000},
        "co_pay_percent": 5,
        "initial_waiting_days": 30,
        "specific_disease_waiting": {"cataract": 24, "hernia": 24},
        "pre_existing_waiting_months": 48,
        "sub_limits": {"cataract": {"amount": 40000, "unit": "per_eye"}},
        "exclusions": ["cosmetic_surgery", "dental_unless_accident"],
    }
)
NO_COPAY = AS.model_copy(update={"co_pay_percent": 0})


def facts(**kw) -> ClaimFacts:
    base = {
        "sum_insured": D(500000),
        "charges": {"room_rent": D(10000), "medicines": D(20000)},
        "room_rate_per_day": D(2000),
    }
    return ClaimFacts(**{**base, **kw})


# ------------------------------------------------------------------------------ calculator


def test_full_cover_has_no_deductions() -> None:
    out = compute_payout(NO_COPAY, facts())
    assert out.claimed_amount == D(30000)
    assert out.payable_amount == D(30000) and out.deductions == []


def test_room_rent_proportionate_deduction() -> None:
    # limit min(2% of 5 lakh, 5000) = 5000; room 8000/day -> factor 0.625 on the three categories
    f = ClaimFacts(
        sum_insured=D(500000),
        room_rate_per_day=D(8000),
        charges={
            "room_rent": D(24000),
            "professional_fees": D(20000),
            "ot_charges": D(10000),
            "medicines": D(12000),
        },
        room_tax=D(1200),
    )
    out = compute_payout(NO_COPAY, f)
    assert out.claimed_amount == D(67200)
    assert [d.rule for d in out.deductions] == ["room_rent_limit"]
    # (25200 - 15750) + (20000 - 12500) + (10000 - 6250) = 9450 + 7500 + 3750
    assert out.deductions[0].amount == D(20700)
    assert out.payable_amount == D(46500)
    assert "8,000/day" in out.deductions[0].description


def test_icu_limit_caps_the_daily_rate() -> None:
    f = facts(
        icu_days=3, icu_rate_per_day=D(14000), charges={"icu": D(42000), "medicines": D(8000)}
    )
    out = compute_payout(NO_COPAY, f)
    assert [d.rule for d in out.deductions] == ["icu_limit"]
    assert out.deductions[0].amount == D(12000)  # (14000 - 10000) x 3
    assert out.payable_amount == D(38000)


def test_sub_limit_per_eye() -> None:
    f = facts(sub_limit_key="cataract", units=1, charges={"surgeon": D(70000)})
    out = compute_payout(NO_COPAY, f)
    assert out.deductions[0].rule == "sub_limits.cataract"
    assert out.payable_amount == D(40000)
    both = compute_payout(
        NO_COPAY, facts(sub_limit_key="cataract", units=2, charges={"surgeon": D(70000)})
    )
    assert both.payable_amount == D(70000) and both.deductions == []  # 2 x 40000 covers it


def test_co_pay() -> None:
    out = compute_payout(AS, facts())
    assert out.deductions[0].rule == "co_pay_percent"
    assert out.deductions[0].amount == D(1500)  # 5% of 30000
    assert out.payable_amount == D(28500)


def test_remaining_sum_insured_cap() -> None:
    out = compute_payout(NO_COPAY, facts(remaining_sum_insured=D(25000)))
    assert [d.rule for d in out.deductions] == ["sum_insured"]
    assert out.deductions[0].amount == D(5000) and out.payable_amount == D(25000)


def test_combination_applies_each_step_on_what_is_left() -> None:
    f = ClaimFacts(
        sum_insured=D(100000),
        remaining_sum_insured=D(60000),
        room_rate_per_day=D(4000),  # limit min(2% of 1 lakh = 2000, 5000) = 2000
        charges={"room_rent": D(20000), "professional_fees": D(30000), "medicines": D(40000)},
    )
    out = compute_payout(AS, f)
    # room: factor 0.5 -> 10000 + 15000 off; running 90000-25000=65000 -> SI cap 60000 (-5000);
    # co-pay 5% of 60000 = 3000
    assert [d.rule for d in out.deductions] == ["room_rent_limit", "sum_insured", "co_pay_percent"]
    assert [d.amount for d in out.deductions] == [D(25000), D(5000), D(3000)]
    assert out.payable_amount == D(57000)
    assert out.claimed_amount - sum(d.amount for d in out.deductions) == out.payable_amount


def test_no_limits_means_nothing_to_deduct() -> None:
    premium = PolicyRules.model_validate({"room_rent_limit": {}, "icu_limit": {}})
    out = compute_payout(premium, facts(room_rate_per_day=D(40000)))
    assert out.deductions == [] and out.payable_amount == D(30000)


# ------------------------------------------------------------------------------ eligibility


def test_eligibility_blockers() -> None:
    start, admit = date(2024, 1, 1), date(2024, 1, 11)
    kinds = lambda **kw: [  # noqa: E731
        b.kind
        for b in check_eligibility(AS, facts(policy_start_date=start, admission_date=admit, **kw))
    ]
    assert kinds() == ["initial_waiting"]
    assert kinds(is_accident=True) == []  # accidents are exempt from the initial waiting period
    assert kinds(is_accident=True, exclusion_tag="dental_unless_accident") == []
    assert "exclusion" in kinds(exclusion_tag="cosmetic_surgery", is_accident=True)
    long_ago = facts(policy_start_date=date(2023, 1, 1), admission_date=date(2024, 6, 1))
    assert (
        check_eligibility(AS, long_ago.model_copy(update={"specific_wait_key": "hernia"}))[0].rule
        == "specific_disease_waiting.hernia"
    )  # 17 months < 24
    ped = long_ago.model_copy(update={"ped_related": True})
    assert check_eligibility(AS, ped)[0].kind == "pre_existing"  # 17 < 48 months


# ------------------------------------------------------------------------------ decision rules


def check(name, result, applies=True, cited=True) -> CoverageCheck:
    clauses = [{"chunk_id": "c1", "section_path": "6.3", "pages": "8"}] if cited else []
    return CoverageCheck.model_validate(
        {
            "check": name,
            "applies": applies,
            "result": result,
            "explanation": "x",
            "cited_clauses": clauses,
        }
    )


def payout(deductions=True):
    f = facts(room_rate_per_day=D(8000), charges={"room_rent": D(24000), "medicines": D(10000)})
    return compute_payout(AS if deductions else NO_COPAY, f if deductions else facts())


def verdict(**kw):
    base = {
        "missing_documents": [],
        "policy_code": "POL-T",
        "coverage": [check("initial_waiting", "pass")],
        "blockers": [],
        "payout": payout(False),
        "risk": RiskAssessment(score=0),
        "review_threshold": 50,
    }
    return decide(**{**base, **kw})


def test_decision_scenarios() -> None:
    assert verdict().decision == "approve"
    assert verdict(payout=payout(True)).decision == "partial_approve"
    d = verdict(missing_documents=["discharge_summary"])
    assert d.decision == "request_documents" and d.payable_amount is None
    assert verdict(policy_code=None).decision == "request_documents"
    high = verdict(
        risk=RiskAssessment(score=60, flags=[RiskFlag(code="x", severity="high", detail="d")])
    )
    assert high.decision == "flag_for_review"
    assert (
        verdict(coverage=[check("room_rent", "unclear", cited=False)]).decision == "flag_for_review"
    )
    assert verdict(coverage=[check("icu_limit", "unclear", applies=False)]).decision == "approve"


def test_blocking_rule_needs_rules_and_wording_to_agree() -> None:
    blocker = check_eligibility(
        AS, facts(policy_start_date=date(2024, 1, 1), admission_date=date(2024, 1, 5))
    )
    agreed = verdict(blockers=blocker, coverage=[check("initial_waiting", "fail")])
    assert agreed.decision == "reject" and agreed.payable_amount == D(0)
    assert agreed.cited_clauses[0].chunk_id == "c1"
    # rules say blocked, the wording check passes: they disagree -> a human decides
    split = verdict(blockers=blocker, coverage=[check("initial_waiting", "pass")])
    assert split.decision == "flag_for_review" and "disagree" in split.facts[0]
    # wording says fail but the rules see no blocker
    assert verdict(coverage=[check("exclusion", "fail")]).decision == "flag_for_review"
    # no policy wording at all (all unclear): rules block, nothing confirms it
    unclear = verdict(blockers=blocker, coverage=[check("initial_waiting", "unclear", cited=False)])
    assert unclear.decision == "flag_for_review"


def test_decision_without_amount_is_flagged() -> None:
    assert verdict(payout=None).decision == "flag_for_review"


# ------------------------------------------------------------------------------ risk rules


SETTINGS = AgentSettings(_env_file=None)


def test_risk_rules_and_score() -> None:
    ext = {"bill_total": "300000", "admission_date": "2025-03-10", "icd10_code": "K35.80"}
    flags = rule_flags(
        extraction=ext,
        issues=[
            {"type": "date_mismatch", "severity": "warning", "detail": "dates differ"},
            {"type": "missing_field", "severity": "warning", "detail": "ignored"},
        ],
        history={
            "days_from_policy_start_to_admission": 12,
            "previous_claims": [{"admission_date": f"2025-0{m}-01"} for m in (1, 2, 3)],
        },
        diagnosis=resolve_diagnosis("K35.80", None),
        settings=SETTINGS,
    )
    codes = {f.code: f.severity for f in flags}
    assert codes == {
        "admission_soon_after_policy_start": "high",
        "bill_exceeds_typical_range": "high",
        "date_mismatch_between_documents": "high",
        "frequent_claims": "medium",
    }
    assert risk_score(flags) == 100  # capped
    assert risk_score([RiskFlag(code="a", severity="medium", detail="")]) == 25
    clean = rule_flags(
        extraction={"bill_total": "70000", "admission_date": "2025-03-10"},
        issues=[],
        history={"days_from_policy_start_to_admission": 400},
        diagnosis=resolve_diagnosis("K35.80", None),
        settings=SETTINGS,
    )
    assert clean == []


def test_llm_flags_are_capped_and_need_justification() -> None:
    base = [RiskFlag(code="llm_odd", severity="high", detail="x")]
    merged = merge_extra_flags(
        base,
        [
            ExtraFlagLLM(code="odd", detail="dup", justification="j"),  # same code as existing
            ExtraFlagLLM(code="No Reason", detail="d", justification=" "),
            ExtraFlagLLM(
                code="weekend admission", detail="Admitted on Sunday", justification="date"
            ),
        ],
    )
    assert [f.code for f in merged] == ["llm_odd", "llm_weekend_admission"]
    assert merged[1].severity == "medium" and merged[1].source == "llm"


# ------------------------------------------------------------------------------ outcome


def approval(action, amount=None, comment=None) -> HumanApproval:
    return HumanApproval(action=action, final_amount=amount, comment=comment, reviewer="r")


def draft(kind, payable=40000, claimed=50000) -> dict:
    return {"decision": kind, "payable_amount": payable, "claimed_amount": claimed}


def test_edited_approval_needs_amount_and_comment() -> None:
    with pytest.raises(ValueError, match="final_amount"):
        approval("edited", comment="why")
    with pytest.raises(ValueError, match="comment"):
        approval("edited", D(100))
    with pytest.raises(ValueError, match="comment"):
        approval("edited", D(100), "   ")
    assert approval("edited", D(100), "adjusted").final_amount == D(100)


def test_final_outcome_mapping() -> None:
    assert final_outcome(draft("approve", 50000), approval("approved")) == (
        ClaimStatus.APPROVED,
        D(50000),
    )
    assert final_outcome(draft("partial_approve"), approval("approved")) == (
        ClaimStatus.PARTIALLY_APPROVED,
        D(40000),
    )
    assert final_outcome(draft("reject", 0), approval("approved"))[0] == ClaimStatus.REJECTED
    assert final_outcome(draft("request_documents", None), approval("approved"))[0] == (
        ClaimStatus.DOCUMENTS_REQUESTED
    )
    assert final_outcome(draft("approve"), approval("rejected")) == (ClaimStatus.REJECTED, D(0))
    assert final_outcome(draft("flag_for_review"), approval("edited", D(30000), "ok")) == (
        ClaimStatus.PARTIALLY_APPROVED,
        D(30000),
    )
    assert final_outcome(draft("flag_for_review"), approval("edited", D(50000), "full"))[0] == (
        ClaimStatus.APPROVED
    )
    for bad in (
        (draft("flag_for_review"), approval("approved")),
        (draft("approve"), approval("edited", D(60000), "more than claimed")),
        (draft("approve"), approval("edited", D(0), "zero")),
        (draft("request_documents", None), approval("edited", D(10), "x")),
    ):
        with pytest.raises(ApprovalError):
            final_outcome(*bad)


# ------------------------------------------------------------------------------ supervisor


def test_supervisor_guards() -> None:
    assert allowed_steps({}) == ["intake"]
    base = {"completed_steps": ["intake"], "policy_code": "POL-T", "missing_documents": []}
    assert allowed_steps(base) == ["coverage", "risk"]
    assert allowed_steps({**base, "completed_steps": ["intake", "coverage"]}) == ["risk"]
    assert allowed_steps({**base, "completed_steps": ["intake", "coverage", "risk"]}) == [
        "decision"
    ]
    # missing documents skip coverage and risk entirely
    assert allowed_steps({**base, "missing_documents": ["bill"]}) == ["decision"]
    assert allowed_steps({**base, "policy_code": None}) == ["decision"]
    assert allowed_steps({**base, "completed_steps": ["intake", "decision"]}) == []
    assert allowed_steps({**base, "errors": [{"node": "x", "error": "y"}]}) == []


def test_invalid_llm_choice_falls_back_to_default_order() -> None:
    allowed = ["coverage", "risk"]
    picked, source = resolve_choice(allowed, SupervisorChoice(next_steps=["risk"], reason="r"))
    assert (picked, source) == (["risk"], "llm")
    for bad in (["decision"], ["coverage", "decision"], []):
        picked, source = resolve_choice(allowed, SupervisorChoice(next_steps=bad, reason="r"))
        assert picked == allowed and source.startswith("default_order")
    assert resolve_choice(allowed, None)[0] == allowed


# ------------------------------------------------------------------------------ citations


def test_citation_validation_drops_unknown_chunk_ids() -> None:
    clauses = known_clauses(
        [
            {"tool": "get_policy_rules", "args": {}, "result": {"rules": {}}},
            {
                "tool": "search_policy",
                "args": {},
                "result": [
                    {
                        "chunk_id": "real-1",
                        "section_path": "6.3 Waiting",
                        "page_start": 8,
                        "page_end": 9,
                    }
                ],
            },
        ]
    )
    assert list(clauses) == ["real-1"]
    out = CoverageLLMOutput(
        checks=[
            CoverageCheckLLM(
                check="initial_waiting",
                applies=True,
                result="fail",
                explanation="e",
                chunk_ids=["real-1", "invented"],
            ),
            CoverageCheckLLM(
                check="exclusion",
                applies=True,
                result="fail",
                explanation="e",
                chunk_ids=["invented"],
            ),
            CoverageCheckLLM(check="co_pay", applies=True, result="pass", explanation="e"),
            CoverageCheckLLM(check="icu_limit", applies=False, result="pass", explanation="n/a"),
            CoverageCheckLLM(
                check="initial_waiting",
                applies=True,
                result="pass",
                explanation="dup",
                chunk_ids=["real-1"],
            ),
        ]
    )
    by_name = {c.check: c for c in validate_checks(out, clauses)}
    assert len(by_name) == 4  # the duplicate check is ignored
    first = by_name["initial_waiting"]
    assert first.result == "fail"
    assert [(c.chunk_id, c.section_path, c.pages) for c in first.cited_clauses] == [
        ("real-1", "6.3 Waiting", "8-9")
    ]
    # fail without any valid citation, and pass with none, both become unclear
    assert by_name["exclusion"].result == "unclear" and by_name["exclusion"].cited_clauses == []
    assert by_name["co_pay"].result == "unclear"
    assert by_name["icu_limit"].result == "pass"  # not applicable: nothing to support


def test_reasons_may_only_use_numbers_they_were_given() -> None:
    given = json.dumps(
        {"facts": ["Claimed Rs 67,200.00, payable Rs 46,500.00."], "deduction": 20700}
    )
    assert reasons_are_grounded(
        ["Payable amount is Rs 46,500.", "Deduction of 20700 applied."], given
    )
    assert not reasons_are_grounded(["Payable amount is Rs 47,000."], given)
    assert reasons_are_grounded(["The room rent is above the limit."], given)


# ------------------------------------------------------------------------------ facts


def test_build_facts_from_extraction() -> None:
    ext = {
        "diagnosis_text": "Senile Cataract",
        "icd10_code": "H25.9",
        "admission_date": "2025-03-01",
        "procedure": "Phacoemulsification, bilateral",
        "room_rent_per_day": "2000",
        "line_items": [
            {"category": "surgeon", "description": "Surgeon Fee", "amount": "30000"},
            {"category": "ot", "amount": "5000"},
        ],
        "taxes": "0",
        "bill_total": "35000",
    }
    f, diagnosis = build_facts(
        ext,
        {},
        {"sum_insured": 500000, "remaining_sum_insured": 450000, "policy_start_date": "2024-01-01"},
        AS,
    )
    assert diagnosis.key == "cataract" and f.sub_limit_key == "cataract" and f.units == 2
    assert f.specific_wait_key == "cataract"
    assert f.charges == {"professional_fees": D(30000), "ot_charges": D(5000)}
    assert f.remaining_sum_insured == D(450000)
    assert build_facts(ext, {}, {}, AS)[0] is None  # unknown sum insured


def test_coverage_output_is_lenient_about_names_and_results() -> None:
    odd = CoverageCheckLLM(check="Room Rent", result="Not Applicable", explanation="x")
    assert (odd.check, odd.result) == ("room_rent", "unclear")
    assert CoverageCheckLLM(check="weird thing", result="PASSED", explanation="x").check == "other"
    assert CoverageCheckLLM(check="co_pay", result=" Fail ", explanation="x").result == "fail"


async def test_invalid_structured_output_is_retried_once_with_the_error() -> None:
    from langchain_core.messages import HumanMessage

    from app.agents.llm import LLMOutputError, ask_structured
    from app.agents.supervisor import SupervisorChoice

    seen: list[list] = []

    class Flaky:
        def __init__(self, fail_times: int) -> None:
            self.fail_times = fail_times

        def with_structured_output(self, schema, **_):
            outer = self

            class Runnable:
                async def ainvoke(self, messages):
                    seen.append(messages)
                    if outer.fail_times > 0:
                        outer.fail_times -= 1
                        return {
                            "raw": None,
                            "parsed": None,
                            "parsing_error": ValueError("bad field"),
                        }
                    return {
                        "raw": None,
                        "parsed": SupervisorChoice(next_steps=["risk"], reason="r"),
                    }

            return Runnable()

    parsed, _ = await ask_structured(Flaky(1), SupervisorChoice, [HumanMessage(content="q")])
    assert parsed.next_steps == ["risk"] and len(seen) == 2
    assert "bad field" in seen[1][-1].content  # the model is shown what was wrong
    with pytest.raises(LLMOutputError):
        await ask_structured(Flaky(5), SupervisorChoice, [HumanMessage(content="q")])


def test_non_professional_items_filed_under_surgeon_are_not_prorated() -> None:
    ext = {
        "room_rent_per_day": "8400",
        "line_items": [
            {"category": "room", "description": "Room Rent - Deluxe Room", "amount": "16800"},
            {"category": "surgeon", "description": "Anaesthetist Fee", "amount": "4695"},
            {"category": "surgeon", "description": "Assistant Surgeon Fee", "amount": "4066"},
            {"category": "surgeon", "description": "Pre-anaesthetic Check-up", "amount": "1330"},
        ],
        "taxes": "0",
        "bill_total": "26891",
    }
    f, _ = build_facts(ext, {}, {"sum_insured": 500000}, AS)
    assert f.charges["professional_fees"] == D(8761) and f.charges["other"] == D(1330)


def test_chunk_ids_written_into_the_explanation_are_still_cited() -> None:
    cid = "42e46d01-16b9-4b87-a280-17eecb1d9a10"
    check = CoverageCheckLLM(
        check="exclusion",
        result="fail",
        explanation=f"Cosmetic surgery is excluded (chunk_id: {cid}).",
    )
    assert check.chunk_ids == [cid] and cid not in check.explanation
    out = CoverageLLMOutput(checks=[check])
    ok = validate_checks(out, {cid: {"section_path": "7 Exclusions", "page_start": 11}})
    assert ok[0].result == "fail" and ok[0].cited_clauses[0].chunk_id == cid
    unknown = validate_checks(out, {})  # an id the tools never returned stays uncitable
    assert unknown[0].result == "unclear"


def test_relevant_checks_follow_the_claim_facts() -> None:
    from app.agents.relevance import relevant_checks

    names = lambda f: [c.check for c in relevant_checks(AS, f)]  # noqa: E731
    assert names(None) == ["initial_waiting"]
    plain = facts(room_rate_per_day=D(2000))  # below the Rs 5,000 limit... but a limit exists
    assert names(plain) == ["initial_waiting", "room_rent", "co_pay"]
    cosmetic = facts(exclusion_tag="cosmetic_surgery", ped_related=True, specific_wait_key="hernia")
    assert names(cosmetic) == [
        "initial_waiting", "specific_disease_waiting", "pre_existing", "exclusion", "room_rent", "co_pay",
    ]  # fmt: skip
    eyes = facts(sub_limit_key="cataract", icu_days=2, room_rate_per_day=D(0))
    assert names(eyes) == ["initial_waiting", "icu_limit", "sub_limit", "co_pay"]
    no_limits = PolicyRules.model_validate({"room_rent_limit": {}, "icu_limit": {}})
    assert [c.check for c in relevant_checks(no_limits, plain)] == ["initial_waiting"]
    assert "cosmetic surgery exclusion" in [
        c.suggested_query for c in relevant_checks(AS, cosmetic)
    ]


def test_validate_checks_keeps_only_the_expected_ones() -> None:
    clauses = {"c1": {"section_path": "6 Waiting Period", "page_start": 8, "page_end": 8}}
    out = CoverageLLMOutput(
        checks=[
            CoverageCheckLLM(
                check="initial_waiting",
                applies=False,
                result="pass",
                explanation="ok",
                chunk_ids=["c1"],
            ),
            CoverageCheckLLM(
                check="icu_limit", applies=True, result="pass", explanation="not needed"
            ),
        ]
    )
    got = {
        c.check: c for c in validate_checks(out, clauses, expected=["initial_waiting", "co_pay"])
    }
    assert set(got) == {"initial_waiting", "co_pay"}  # icu_limit dropped, co_pay added
    assert got["initial_waiting"].applies and got["initial_waiting"].result == "pass"
    assert got["co_pay"].result == "unclear" and got["co_pay"].applies


def test_clause_refs_are_short_stable_and_resolve_back_to_chunk_ids() -> None:
    from app.agents.nodes.coverage import ClauseRefs

    refs = ClauseRefs()
    first = json.loads(
        refs.render(
            "search_policy",
            [{"chunk_id": "id-a", "section_path": "6", "content": "x"}, {"chunk_id": "id-b"}],
        )
    )
    again = json.loads(refs.render("search_policy", [{"chunk_id": "id-b"}, {"chunk_id": "id-c"}]))
    assert [c["ref"] for c in first] == ["C1", "C2"] and "chunk_id" not in first[0]
    assert [c["ref"] for c in again] == ["C2", "C3"]  # the same clause keeps its ref
    assert [refs.resolve(t) for t in ("C1", "[c3]", " C2.", "C9", "id-a")] == [
        "id-a", "id-c", "id-b", "C9", "id-a",
    ]  # fmt: skip
    assert json.loads(refs.render("get_policy_rules", {"rules": {}})) == {"rules": {}}

    out = CoverageLLMOutput(
        checks=[
            CoverageCheckLLM(check="co_pay", result="pass", explanation="e", chunk_ids=["C2", "C9"])
        ]
    )
    clauses = {"id-b": {"section_path": "3. DEFINITIONS", "page_start": 2, "page_end": 3}}
    got = validate_checks(out, clauses, refs=refs)[0]
    assert got.result == "pass" and [c.chunk_id for c in got.cited_clauses] == ["id-b"]


def test_amount_checks_follow_the_calculation() -> None:
    from app.agents.relevance import reconcile_amount_checks

    deducting = compute_payout(
        AS, facts(room_rate_per_day=D(8000), charges={"room_rent": D(24000), "medicines": D(10000)})
    )
    clean = compute_payout(AS, facts(room_rate_per_day=D(2000)))
    wrongly_failed = check("room_rent", "fail")
    wrongly_passed = check("room_rent", "pass")
    kept = check("room_rent", "unclear", cited=False)
    other = check("initial_waiting", "pass")

    got = reconcile_amount_checks([wrongly_failed, kept, other], clean)
    assert [c.result for c in got] == ["pass", "unclear", "pass"]
    assert "no deduction" in got[0].explanation and got[0].cited_clauses  # citations are kept
    got = reconcile_amount_checks([wrongly_passed, check("room_rent", "partial")], deducting)
    assert [c.result for c in got] == ["partial", "partial"]
    assert "deduction" in got[0].explanation and got[1].explanation == "x"  # already consistent
    assert reconcile_amount_checks([wrongly_failed], None)[0].result == "fail"  # nothing to compare
