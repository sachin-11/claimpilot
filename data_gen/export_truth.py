"""Write data/golden/extraction_truth.json: the TRUE field values of every generated claim.

    python -m data_gen.export_truth --count 40 --seed 42 --out data

The generator is deterministic (same seed -> same plan), and planning does not involve the LLM, so
this re-plans the claims and writes ONLY the truth file: no PDF, scan, claim form or golden
expected.json is touched. Before writing, every re-planned claim is checked against the files
already on disk (claim_form.json, expected.json); any difference means the seed / count / policy
rules changed since the data was generated, and nothing is written.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from data_gen.config import load_policy_rules, load_typical_costs
from data_gen.models import PlannedClaim
from data_gen.planner import plan_claims

logger = logging.getLogger("data_gen.export_truth")
app = typer.Typer(add_completion=False, help="Export extraction ground truth for the intake eval.")


def expected_consistency_issues(planned: PlannedClaim) -> list[str]:
    """Issue types a correct intake should report for this claim (facts visible in the documents)."""
    spec = planned.spec
    issues: list[str] = []
    if spec.omit_discharge_summary:
        issues.append("missing_document:discharge_summary")
    if (spec.summary_admission_date, spec.summary_discharge_date) != (
        spec.admission_date,
        spec.discharge_date,
    ) and not spec.omit_discharge_summary:
        issues.append("date_mismatch")
    if any(line.is_duplicate for line in spec.bill_lines):
        issues.append("duplicate_line_items")
    if spec.summary_discharge_date < spec.summary_admission_date:
        issues.append("discharge_before_admission")
    if "discharge_summary.admission_date" in spec.blank_fields:
        issues.append("missing_field:admission_date")
    if spec.policy_number is None:
        issues.append("missing_field:policy_number")
    if "hospital_bill.total_amount" in spec.blank_fields:
        issues.append("missing_field:bill_total")
    return sorted(set(issues))


def truth_record(planned: PlannedClaim) -> dict:
    spec = planned.spec
    blank_total = "hospital_bill.total_amount" in spec.blank_fields
    blank_admission = "discharge_summary.admission_date" in spec.blank_fields
    return {
        "claim_id": spec.claim_id,
        "scenario": spec.scenario.value,
        "subtype": spec.subtype,
        "has_scanned_copy": planned.golden.has_scanned_copy,
        "expected_missing_documents": [
            m for m in planned.golden.expected_missing_documents if "." not in m
        ],
        "expected_consistency_issues": expected_consistency_issues(planned),
        # fields as a correct extraction should return them (the BILL is the reference for dates:
        # the discharge summary may legitimately differ or be blank, that is what is tested)
        "fields": {
            "policy_number": spec.policy_number,
            "patient_name": spec.patient.name,
            "hospital_name": spec.hospital.name,
            "admission_date": spec.admission_date.isoformat(),
            "discharge_date": spec.discharge_date.isoformat(),
            "diagnosis_text": spec.diagnosis_name,
            "icd10_code": spec.icd10,
            "bill_subtotal": spec.bill_subtotal,
            "taxes": spec.gst_on_room,
            "bill_total": None if blank_total else spec.bill_total,
            "claimed_amount": spec.claimed_amount,
            "line_item_count": len(spec.bill_lines),
            "length_of_stay_days": (spec.discharge_date - spec.admission_date).days,
        },
        "line_items": [
            {
                "description": line.description,
                "category": line.category,
                "qty": line.qty,
                "rate": line.rate,
                "amount": line.amount,
                "is_duplicate": line.is_duplicate,
            }
            for line in spec.bill_lines
        ],
        "documents": {
            "discharge_summary_present": not spec.omit_discharge_summary,
            "summary_admission_date": None
            if blank_admission or spec.omit_discharge_summary
            else spec.summary_admission_date.isoformat(),
            "summary_discharge_date": None
            if spec.omit_discharge_summary
            else spec.summary_discharge_date.isoformat(),
            "bill_total_blank": blank_total,
        },
        # personal data in the documents: used to measure PII leakage after masking
        "pii": {
            "name": spec.patient.name,
            "phone": spec.patient.phone,
            "email": spec.patient.email,
            "address": spec.patient.address,
            "uhid": spec.patient.uhid,
        },
    }


def check_against_disk(planned: list[PlannedClaim], out: Path) -> list[str]:
    """Differences between the re-planned claims and the files that already exist."""
    problems: list[str] = []
    expected = {r["claim_id"]: r for r in json.loads((out / "golden" / "expected.json").read_text("utf-8"))}
    for p in planned:
        spec = p.spec
        form_path = out / "claims" / spec.claim_id / "claim_form.json"
        if not form_path.exists():
            problems.append(f"{spec.claim_id}: no claim_form.json on disk")
            continue
        form = json.loads(form_path.read_text("utf-8"))
        checks = {
            "claimed_amount": (form["claimed_amount"], spec.claimed_amount),
            "diagnosis": (form["diagnosis"], spec.diagnosis_name),
            "policy_number": (form["policy_number"], spec.policy_number),
            "admission_date": (form["admission_date"], spec.admission_date.isoformat()),
            "patient": (form["claimant"]["name"], spec.patient.name),
        }
        for name, (on_disk, planned_value) in checks.items():
            if on_disk != planned_value:
                problems.append(f"{spec.claim_id}: {name} on disk {on_disk!r} != planned {planned_value!r}")
        golden = expected.get(spec.claim_id)
        if golden is None or golden["icd10"] != spec.icd10 or golden["claimed_amount"] != spec.claimed_amount:
            problems.append(f"{spec.claim_id}: expected.json differs from the re-planned claim")
    return problems


@app.command()
def main(
    count: Annotated[int, typer.Option(help="Same value as used for generation")] = 40,
    seed: Annotated[int, typer.Option(help="Same seed as used for generation")] = 42,
    out: Annotated[Path, typer.Option(help="Output root containing claims/ and golden/")] = Path("data"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    planned = plan_claims(count, seed, load_policy_rules(None), load_typical_costs(None), None)
    scan_ids = {p.parent.name for p in (out / "claims").glob("*/*_scanned.png")}
    for p in planned:  # same flag the generator stored in expected.json
        p.golden.has_scanned_copy = p.spec.claim_id in scan_ids

    problems = check_against_disk(planned, out)
    if problems:
        sys.stderr.write("The re-planned claims do not match the generated data:\n")
        sys.stderr.write("\n".join(f"  {p}" for p in problems[:20]) + "\n")
        sys.stderr.write("Nothing written. Use the same --count/--seed/policy rules as generation.\n")
        raise typer.Exit(1)

    records = [truth_record(p) for p in planned]
    target = out / "golden" / "extraction_truth.json"
    target.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d truth records to %s", len(records), target)


if __name__ == "__main__":
    app()
