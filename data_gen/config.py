"""Load and validate policy_rules.yaml and typical_costs.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

PACKAGE_DIR = Path(__file__).parent
DEFAULT_POLICY_RULES = PACKAGE_DIR / "policy_rules.yaml"
DEFAULT_TYPICAL_COSTS = PACKAGE_DIR / "typical_costs.yaml"


class PerDayLimit(BaseModel):
    """Per-day cap: min(percent of sum insured, absolute max). Both null = no limit."""

    percent_of_si_per_day: float | None = None
    max_per_day: int | None = None

    def limit_per_day(self, sum_insured: int) -> int | None:
        candidates: list[float] = []
        if self.percent_of_si_per_day is not None:
            candidates.append(sum_insured * self.percent_of_si_per_day / 100)
        if self.max_per_day is not None:
            candidates.append(self.max_per_day)
        return round(min(candidates)) if candidates else None


class SubLimit(BaseModel):
    amount: int
    unit: Literal["per_eye", "per_admission"] = "per_admission"


class MaternityRule(BaseModel):
    covered: bool = False
    waiting_months: int = 0


class Policy(BaseModel):
    policy_id: str
    name: str
    source_pdf: str
    sum_insured_options: list[int]
    room_rent_limit: PerDayLimit = Field(default_factory=PerDayLimit)
    icu_limit: PerDayLimit = Field(default_factory=PerDayLimit)
    co_pay_percent: float = 0
    initial_waiting_days: int = 0
    specific_disease_waiting: dict[str, int] = Field(default_factory=dict)
    pre_existing_waiting_months: int = 0
    sub_limits: dict[str, SubLimit] = Field(default_factory=dict)
    exclusions: list[str] = Field(default_factory=list)
    maternity: MaternityRule = Field(default_factory=MaternityRule)
    proportionate_categories: list[str] = Field(
        default_factory=lambda: ["room_rent", "professional_fees", "ot_charges"]
    )


class PolicyRules(BaseModel):
    policies: list[Policy]

    def get(self, policy_id: str) -> Policy:
        for policy in self.policies:
            if policy.policy_id == policy_id:
                return policy
        raise KeyError(f"Unknown policy_id: {policy_id}")


class Diagnosis(BaseModel):
    key: str
    name: str
    icd10: str
    procedure: str | None = None
    kind: Literal["medical", "surgical", "daycare"]
    stay_days: tuple[int, int]
    cost_range: tuple[int, int]
    icu_probability: float = 0.0
    accident: bool = False
    is_maternity: bool = False
    exclusion_tag: str | None = None
    specific_wait_key: str | None = None
    sub_limit_key: str | None = None
    ped_condition: str | None = None
    implant: str | None = None
    age_range: tuple[int, int] = (22, 70)
    gender: Literal["any", "female"] = "any"
    tests: list[str] = Field(default_factory=list)


class TypicalCosts(BaseModel):
    diagnoses: list[Diagnosis]

    def get(self, key: str) -> Diagnosis:
        for diagnosis in self.diagnoses:
            if diagnosis.key == key:
                return diagnosis
        raise KeyError(f"Unknown diagnosis key: {key}")


def load_policy_rules(path: Path | None = None) -> PolicyRules:
    with (path or DEFAULT_POLICY_RULES).open(encoding="utf-8") as f:
        return PolicyRules.model_validate(yaml.safe_load(f))


def load_typical_costs(path: Path | None = None) -> TypicalCosts:
    with (path or DEFAULT_TYPICAL_COSTS).open(encoding="utf-8") as f:
        return TypicalCosts.model_validate(yaml.safe_load(f))
