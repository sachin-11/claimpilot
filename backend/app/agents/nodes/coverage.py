"""Coverage agent: a ReAct subgraph over the policy_docs tools that checks the claim against the
policy wording and cites clauses. Its findings are validated before they reach the state."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agents.calculator import (
    ClaimFacts,
    PayoutResult,
    PolicyRules,
    compute_payout,
    months_between,
)
from app.agents.catalog import resolve_diagnosis
from app.agents.deps import AgentDeps
from app.agents.events import forward_events
from app.agents.facts import build_facts
from app.agents.llm import ask_structured
from app.agents.prompts import load_prompt
from app.agents.react import ToolRecord, run_react
from app.agents.relevance import (
    RelevantCheck,
    reconcile_amount_checks,
    reconcile_waiting_checks,
    relevant_checks,
)
from app.agents.state import CitedClause, ClaimState, CoverageCheck
from app.agents.tools import toolbox_for
from app.core.redaction import redact

logger = logging.getLogger(__name__)


UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
CHECK_NAMES = {
    "initial_waiting", "specific_disease_waiting", "pre_existing", "exclusion",
    "room_rent", "icu_limit", "sub_limit", "co_pay", "other",
}  # fmt: skip
RESULT_SYNONYMS = {
    "pass": "pass", "passed": "pass", "ok": "pass", "covered": "pass", "met": "pass",
    "fail": "fail", "failed": "fail", "excluded": "fail", "not covered": "fail",
    "partial": "partial", "partially": "partial", "partially covered": "partial",
}  # fmt: skip


class ClauseRefs:
    """Short, stable labels (C1, C2, ...) for the clauses search_policy returns in one run.

    Models copy a 36-character UUID unreliably and a short label reliably (the same lesson as the
    RAG answerer). The model sees and cites refs; the code maps them back to real chunk ids, so
    the citation check still runs against the tool results."""

    def __init__(self) -> None:
        self.by_chunk: dict[str, str] = {}
        self.by_ref: dict[str, str] = {}

    def render(self, tool: str, result: Any) -> str:
        """Tool result as text for the model: clauses carry `ref` instead of `chunk_id`."""
        if tool != "search_policy" or not isinstance(result, list):
            return json.dumps(result, default=str)
        shown = []
        for clause in result:
            chunk_id = str(clause.get("chunk_id"))
            ref = self.by_chunk.setdefault(chunk_id, f"C{len(self.by_chunk) + 1}")
            self.by_ref[ref] = chunk_id
            shown.append({"ref": ref, **{k: v for k, v in clause.items() if k != "chunk_id"}})
        return json.dumps(shown, default=str)

    def resolve(self, token: str) -> str:
        """A ref ("C2", "[c2]") -> chunk id; anything else (e.g. a real UUID) is returned as is."""
        match = re.fullmatch(r"\W*[Cc]\s*(\d+)\W*", token)
        return self.by_ref.get(f"C{match.group(1)}", token) if match else token


def _require_wording_fields(schema: dict[str, Any]) -> None:
    """The model must always fill the period fields (null for checks without one): optional fields
    are skipped by the model, which would leave every waiting check unverified."""
    required = set(schema.get("required", [])) | {"wording_value", "wording_unit"}
    schema["required"] = sorted(required)


class CoverageCheckLLM(BaseModel):
    """Lenient on purpose: an off-list check name becomes `other` and an off-list result becomes
    `unclear` (the safe, human-reviewed outcome) instead of failing the whole run."""

    model_config = ConfigDict(json_schema_extra=_require_wording_fields)

    check: str
    applies: bool = True
    result: str
    explanation: str = Field(default="", description="1-2 sentences")
    chunk_ids: list[str] = Field(
        default_factory=list,
        description='refs of the clauses you rely on, as in the search results: ["C1", "C3"]',
    )
    wording_value: float | None = Field(
        default=None,
        description="waiting checks only: the waiting period the clause states, e.g. 24",
    )
    wording_unit: str | None = Field(
        default=None, description="waiting checks only: days | months | years"
    )

    @model_validator(mode="after")
    def move_ids_out_of_the_explanation(self) -> "CoverageCheckLLM":
        """Models sometimes write "(chunk_id: <uuid>)" into the explanation, not into chunk_ids.
        Collect them (they are still validated against this run's tool results), tidy the text."""
        found = UUID_RE.findall(self.explanation)
        if found:
            self.chunk_ids = list(dict.fromkeys([*self.chunk_ids, *found]))
            text = re.sub(
                r"\(?\s*(?:chunk[_ ]?ids?\s*:?\s*)?" + UUID_RE.pattern + r"\s*\)?",
                "",
                self.explanation,
            )
            self.explanation = re.sub(r"\s{2,}", " ", text).strip(" ,;")
        return self

    @field_validator("check", mode="after")
    @classmethod
    def known_check(cls, value: str) -> str:
        value = value.strip().lower().replace(" ", "_").replace("-", "_")
        return value if value in CHECK_NAMES else "other"

    @field_validator("result", mode="after")
    @classmethod
    def known_result(cls, value: str) -> str:
        return RESULT_SYNONYMS.get(value.strip().lower(), "unclear")


class CoverageLLMOutput(BaseModel):
    checks: list[CoverageCheckLLM]


def claim_summary(state: ClaimState) -> dict[str, Any]:
    """Facts for the coverage prompt: no names, contact details or free-text identifiers."""
    ext = state.get("extraction") or {}
    history = state.get("customer_history") or {}
    diagnosis = resolve_diagnosis(ext.get("icd10_code"), ext.get("diagnosis_text"))
    by_category: dict[str, float] = {}
    for item in ext.get("line_items") or []:
        key = str(item.get("category"))
        by_category[key] = by_category.get(key, 0.0) + float(item["amount"])
    months = None
    if history.get("policy_start_date") and ext.get("admission_date"):
        from datetime import date

        months = months_between(
            date.fromisoformat(history["policy_start_date"]),
            date.fromisoformat(ext["admission_date"]),
        )
    return {
        "policy_code": state.get("policy_code"),
        "diagnosis": redact(str(ext.get("diagnosis_text") or "")),
        "icd10": ext.get("icd10_code"),
        "diagnosis_key": diagnosis.key if diagnosis else None,
        "procedure": redact(str(ext.get("procedure") or "")),
        "admission_date": ext.get("admission_date"),
        "discharge_date": ext.get("discharge_date"),
        "policy_start_date": history.get("policy_start_date"),
        "days_from_policy_start_to_admission": history.get("days_from_policy_start_to_admission"),
        "months_from_policy_start_to_admission": months,
        "sum_insured": history.get("sum_insured"),
        "room_type": ext.get("room_type"),
        "room_rent_per_day": ext.get("room_rent_per_day"),
        "icu_days": ext.get("icu_days"),
        "bill_by_category": by_category,
        "bill_total": ext.get("bill_total"),
        "declared_pre_existing_conditions": (state.get("claim_facts") or {}).get(
            "pre_existing_conditions"
        ),
        "is_accident": (state.get("claim_facts") or {}).get("is_accident"),
    }


def _pages(start: Any, end: Any) -> str | None:
    if start is None:
        return None
    return str(start) if end in (None, start) else f"{start}-{end}"


def known_clauses(records: list[ToolRecord]) -> dict[str, dict[str, Any]]:
    """Every clause search_policy returned in this run, by chunk_id. The only citable ones."""
    found: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["tool"] != "search_policy":
            continue
        for clause in record["result"] or []:
            if isinstance(clause, dict) and clause.get("chunk_id"):
                found[str(clause["chunk_id"])] = clause
    return found


def validate_checks(
    output: CoverageLLMOutput,
    clauses: dict[str, dict[str, Any]],
    expected: list[str] | None = None,
    refs: ClauseRefs | None = None,
) -> list[CoverageCheck]:
    """Citations must come from this run's tool results; section and pages come from OUR records,
    not from the model. A conclusion without a supporting clause becomes `unclear`.

    With `expected` (the checks relevant to this claim) only those are kept, all of them count as
    applicable, and one the model skipped is added as `unclear`."""
    checks: dict[str, CoverageCheck] = {}
    for raw in output.checks:
        if raw.check in checks or (expected is not None and raw.check not in expected):
            continue
        if expected is not None:
            raw.applies = True
        ids = [refs.resolve(token) if refs else token for token in raw.chunk_ids]
        cited = [
            CitedClause(
                chunk_id=cid,
                section_path=clauses[cid].get("section_path"),
                pages=_pages(clauses[cid].get("page_start"), clauses[cid].get("page_end")),
            )
            for cid in dict.fromkeys(ids)
            if cid in clauses
        ]
        dropped = len(set(ids)) - len(cited)
        if dropped:
            logger.warning("coverage_citation_dropped", extra={"check": raw.check, "n": dropped})
        result, explanation = raw.result, raw.explanation
        if raw.applies and result != "unclear" and not cited:
            result = "unclear"
            explanation = (
                f"{explanation} (No supporting policy clause was found, so this is unclear.)"
            )
        checks[raw.check] = CoverageCheck(
            check=raw.check,
            applies=raw.applies,
            result=result,
            explanation=explanation,
            cited_clauses=cited,
        )
    for name in expected or []:
        if name not in checks:
            checks[name] = CoverageCheck(
                check=name,  # type: ignore[arg-type]
                applies=True,
                result="unclear",
                explanation="This check was not assessed.",
            )
    return list(checks.values())


def checks_to_evaluate(
    state: ClaimState, rules: dict[str, Any]
) -> tuple[list[RelevantCheck], PayoutResult | None, ClaimFacts | None, PolicyRules]:
    """The checks relevant to this claim and the calculation, from structured facts (no LLM)."""
    policy_rules = PolicyRules.model_validate(rules)
    facts, _ = build_facts(
        state.get("extraction") or {},
        state.get("claim_facts") or {},
        state.get("customer_history") or {},
        policy_rules,
    )
    payout = compute_payout(policy_rules, facts) if facts is not None else None
    return relevant_checks(policy_rules, facts), payout, facts, policy_rules


def make_coverage_node(deps: AgentDeps):  # type: ignore[no-untyped-def]
    async def coverage(state: ClaimState) -> dict[str, Any]:
        tools = await toolbox_for(deps.tools, "coverage")
        code = state["policy_code"]

        rules_out = await tools.call("get_policy_rules", {"policy_code": code})
        rules = rules_out["rules"]

        summary = claim_summary(state)
        wanted, payout, facts, policy_rules = checks_to_evaluate(state, rules)
        refs = ClauseRefs()
        messages: list[BaseMessage] = [
            SystemMessage(content=load_prompt("coverage")),
            HumanMessage(
                content="Claim facts:\n"
                + json.dumps(summary, default=str, indent=1)
                + "\n\npolicy_rules:\n"
                + json.dumps(rules, default=str)
                + "\n\nchecks_to_evaluate (exactly these; search with the suggested query first):\n"
                + json.dumps([c.__dict__ for c in wanted], indent=1)
            ),
        ]
        model = deps.chat_model()
        with_tools = model.bind_tools(tools.langchain_tools())
        with forward_events():  # tool_called events from the subgraph reach the run's stream
            messages, records, exhausted = await run_react(
                with_tools,
                tools,
                messages,
                max_tool_calls=deps.settings.coverage_max_tool_calls,
                render=refs.render,
            )
        closing = (
            "Tool budget used up. " if exhausted else ""
        ) + "Now give your final assessment: one entry per check, with chunk_ids you were shown."
        output, model_name = await ask_structured(
            model, CoverageLLMOutput, [*messages, HumanMessage(content=closing)]
        )
        checks = reconcile_amount_checks(
            reconcile_waiting_checks(
                validate_checks(
                    output, known_clauses(records), expected=[c.check for c in wanted], refs=refs
                ),
                {c.check: (c.wording_value, c.wording_unit) for c in output.checks},
                policy_rules,
                facts,
            ),
            payout,
        )
        return {
            "policy_rules": rules,
            "coverage": [c.model_dump(mode="json") for c in checks],
            "models_used": {"coverage": model_name or "unknown"},
            "completed_steps": ["coverage"],
        }

    return coverage
