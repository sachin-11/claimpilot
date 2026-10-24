"""Write claim folders and the golden dataset to disk."""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
from pathlib import Path

from data_gen.llm_text import get_narrative
from data_gen.models import ClaimForm, GoldenRecord, PlannedClaim
from data_gen.pdf_renderer import render_bill, render_summary
from data_gen.scan_simulator import scan_pdf

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "claim_id", "scenario_type", "subtype", "policy_id", "diagnosis", "icd10", "claimed_amount",
    "expected_payable_amount", "expected_decision", "expected_policy_rule", "expected_risk_flags",
    "expected_missing_documents", "has_scanned_copy",
]  # fmt: skip


def clean_previous_output(out_dir: Path) -> None:
    """Remove stale CLM-* folders (only generated artifacts; policies are never touched)."""
    claims_dir = out_dir / "claims"
    if claims_dir.exists():
        for child in claims_dir.glob("CLM-*"):
            if child.is_dir():
                shutil.rmtree(child)


def write_claim(
    planned: PlannedClaim, out_dir: Path, *, use_llm: bool, model: str, seed: int, scan: bool
) -> None:
    spec = planned.spec
    folder = out_dir / "claims" / spec.claim_id
    folder.mkdir(parents=True, exist_ok=True)

    form = ClaimForm.from_spec(spec)
    (folder / "claim_form.json").write_text(form.model_dump_json(indent=2), encoding="utf-8")

    bill_path = folder / "hospital_bill.pdf"
    render_bill(spec, bill_path)

    summary_path = folder / "discharge_summary.pdf"
    if not spec.omit_discharge_summary:
        narrative = get_narrative(spec, use_llm=use_llm, model=model, cache_dir=out_dir / ".llm_cache", seed=seed)
        render_summary(spec, narrative, summary_path)

    if scan:
        rng = random.Random(f"{seed}:{spec.claim_id}:scan")
        scan_pdf(bill_path, folder / "hospital_bill_scanned.png", rng)
        if summary_path.exists():
            scan_pdf(summary_path, folder / "discharge_summary_scanned.png", rng)


def write_golden(records: list[GoldenRecord], out_dir: Path) -> None:
    golden_dir = out_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump(mode="json") for r in records]
    (golden_dir / "expected.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (golden_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in payload:
            row = {k: row[k] for k in CSV_COLUMNS}
            row["expected_risk_flags"] = ";".join(row["expected_risk_flags"])
            row["expected_missing_documents"] = ";".join(row["expected_missing_documents"])
            writer.writerow(row)
