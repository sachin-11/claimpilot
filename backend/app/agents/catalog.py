"""Diagnosis catalogue: maps a diagnosis (ICD-10 / name) to policy-rule keys and a typical bill
range. Data lives in config/diagnoses.yaml (own copy; app code never imports data_gen)."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).parent / "config" / "diagnoses.yaml"


@dataclass(frozen=True)
class Diagnosis:
    key: str
    name: str
    icd10: str
    cost_min: int
    cost_max: int
    specific_wait_key: str | None = None
    sub_limit_key: str | None = None
    exclusion_tag: str | None = None
    accident: bool = False
    is_maternity: bool = False
    ped_condition: str | None = None


@lru_cache
def load_catalog() -> tuple[Diagnosis, ...]:
    raw = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    return tuple(
        Diagnosis(
            key=d["key"],
            name=d["name"],
            icd10=d["icd10"],
            cost_min=d["cost_range"][0],
            cost_max=d["cost_range"][1],
            specific_wait_key=d.get("specific_wait_key"),
            sub_limit_key=d.get("sub_limit_key"),
            exclusion_tag=d.get("exclusion_tag"),
            accident=bool(d.get("accident")),
            is_maternity=bool(d.get("is_maternity")),
            ped_condition=d.get("ped_condition"),
        )
        for d in raw["diagnoses"]
    )


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def resolve_diagnosis(icd10: str | None, diagnosis_text: str | None) -> Diagnosis | None:
    """Exact ICD-10 first, then the diagnosis name. None when the catalogue does not know it."""
    catalog = load_catalog()
    if icd10:
        code = icd10.strip().upper()
        for d in catalog:
            if d.icd10.upper() == code:
                return d
    text = _norm(diagnosis_text)
    if text:
        for d in catalog:
            if _norm(d.name) == text or _norm(d.name) in text or text in _norm(d.name):
                return d
    return None
