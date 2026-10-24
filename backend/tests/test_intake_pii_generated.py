"""PII masking measured on ALL generated claims (real PDFs, ground truth from the generator).

Leak      = a true personal value (name, phone, e-mail, address, UHID) is still in the masked text.
Over-mask = something extraction needs (line items, policy number, dates, amounts, diagnosis,
            ICD code, hospital) disappeared from the masked text.
"""

import json
from datetime import date

import pymupdf
import pytest
from intake_fakes import FakeOCR, intake_settings

from app.core.config import get_settings
from app.intake.pii import PIIMasker
from app.intake.text_extract import extract_pdf

DATA = get_settings().data_dir
TRUTH_PATH = DATA / "golden" / "extraction_truth.json"
pytestmark = pytest.mark.skipif(
    not TRUTH_PATH.exists(), reason="data/golden/extraction_truth.json not available"
)
SETTINGS = intake_settings()


def truth_records() -> list[dict]:
    return json.loads(TRUTH_PATH.read_text("utf-8"))


def pages_of(claim_id: str, name: str) -> list[str]:
    path = DATA / "claims" / claim_id / name
    if not path.exists():
        return []
    return [p.text for p in extract_pdf(path.read_bytes(), SETTINGS, FakeOCR())]


def printed_dates(iso: str | None) -> list[str]:
    if not iso:
        return []
    d = date.fromisoformat(iso)
    return [d.strftime("%d-%m-%Y")]


def money(value: int) -> str:
    return f"{value:,}"


def run(use_known: bool) -> tuple[list[str], list[str]]:
    leaks: list[str] = []
    overmasked: list[str] = []
    for t in truth_records():
        claim_id, pii = t["claim_id"], t["pii"]
        known = (
            {
                "PERSON": [pii["name"]],
                "PHONE": [pii["phone"]],
                "EMAIL_ADDRESS": [pii["email"]],
                "ADDRESS": [pii["address"]],
            }
            if use_known
            else {}
        )
        masker = PIIMasker(known)
        for doc in ("hospital_bill.pdf", "discharge_summary.pdf"):
            pages = pages_of(claim_id, doc)
            if not pages:
                continue
            original = "\n".join(pages)
            masked = "\n".join(masker.mask(p) for p in pages)
            for label, value in pii.items():
                if value in original and value in masked:
                    leaks.append(f"{claim_id} {doc}: {label} {value!r} not masked")

            f = t["fields"]
            needed = {
                "policy": f["policy_number"]
                if f["policy_number"] and f["policy_number"] in original
                else None,
                "hospital": f["hospital_name"],
                "diagnosis": f["diagnosis_text"] if f["diagnosis_text"] in original else None,
                "icd": f["icd10_code"] if f["icd10_code"] in original else None,
            }
            for date_value in (f["admission_date"], f["discharge_date"]):
                for printed in printed_dates(date_value):
                    if printed in original:
                        needed[f"date {printed}"] = printed
            if doc == "hospital_bill.pdf":
                for item in t["line_items"]:
                    needed[f"line item {item['description']}"] = item["description"]
                if f["bill_total"] and money(f["bill_total"]) in original:
                    needed["bill total"] = money(f["bill_total"])
            for label, value in needed.items():
                if value and value in original and value not in masked:
                    overmasked.append(f"{claim_id} {doc}: {label} {value!r} was masked")
    return leaks, overmasked


def test_no_personal_value_leaks_and_nothing_needed_is_masked_when_claim_form_values_are_known() -> (
    None
):
    leaks, overmasked = run(use_known=True)

    assert leaks == [], "personal data reached the LLM prompt:\n" + "\n".join(leaks[:15])
    assert overmasked == [], "extraction inputs were masked:\n" + "\n".join(overmasked[:15])


def test_detection_alone_never_masks_what_extraction_needs_and_catches_phones_and_emails() -> None:
    """Uploaded claims have no claim form to learn names from: names rely on NER / labels / titles,
    so some name leakage is possible there, but it must not damage the extraction inputs."""
    leaks, overmasked = run(use_known=False)

    assert overmasked == [], "extraction inputs were masked:\n" + "\n".join(overmasked[:15])
    structural = [m for m in leaks if "phone" in m or "email" in m or "uhid" in m]
    assert structural == [], "\n".join(structural[:15])
    name_leaks = [m for m in leaks if " name " in m or "name '" in m]
    assert len(name_leaks) <= 4, (
        f"{len(name_leaks)} name leaks without the claim form:\n" + "\n".join(name_leaks[:10])
    )


def test_the_generated_documents_really_contain_the_personal_data_we_test_for() -> None:
    t = truth_records()[0]
    text = "\n".join(pages_of(t["claim_id"], "hospital_bill.pdf"))
    assert t["pii"]["name"] in text and t["pii"]["phone"] in text
    assert pymupdf.open(DATA / "claims" / t["claim_id"] / "hospital_bill.pdf").page_count >= 1
