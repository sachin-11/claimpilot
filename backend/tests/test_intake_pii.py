"""PII masking: names / phones / Aadhaar masked and restored; policy numbers and amounts untouched."""

import json
import logging

import pytest

from app.core.logging import setup_logging
from app.core.redaction import PIIRedactingFilter, redact
from app.intake.pii import PIIMasker

SAMPLE = """Prakash Nursing & Maternity Home
55/350 Goyal Road, Indore
Phone: 3806576302 | GSTIN: 31PLCZA4438Z1ZR | Reg. No: REG/4714/2006
Patient Name
Suhani Bala
Patient Phone 07649716752  Email: suhani15@example.com  UHID UH6627725
Aadhaar 2345 6789 0123   PAN ABCDE1234F
Address: 48, Bhasin Path, Junagadh 828219
Treating Doctor Dr. Veer Mand
Policy POL-PREM/2025/6045547  Bill No. BILL/2025/36987
Date of Admission 12-11-2025  Discharge 2025-11-14
Diagnosis Normal Vaginal Delivery (ICD-10: O80)
Sub Total 77,317  GST 600  TOTAL 77,917  Claimed Rs. 1,50,000/-"""

PII_VALUES = [
    "Suhani Bala",
    "Suhani",
    "07649716752",
    "suhani15@example.com",
    "2345 6789 0123",
    "ABCDE1234F",
    "Bhasin Path",
    "Veer Mand",
    "UH6627725",
]
MUST_SURVIVE = [
    "POL-PREM/2025/6045547",
    "BILL/2025/36987",
    "12-11-2025",
    "2025-11-14",
    "O80",
    "77,317",
    "77,917",
    "Rs. 1,50,000/-",
    "Prakash Nursing & Maternity Home",
    "31PLCZA4438Z1ZR",
    "REG/4714/2006",
    "Normal Vaginal Delivery",
]  # (the hospital's own phone is masked too: any 10-digit number is treated as a phone)


@pytest.fixture(scope="module")
def known() -> dict[str, list[str]]:
    return {
        "PERSON": ["Suhani Bala"],
        "PHONE": ["07649716752"],
        "EMAIL_ADDRESS": ["suhani15@example.com"],
        "ADDRESS": ["48, Bhasin Path, Junagadh 828219"],
    }


@pytest.mark.parametrize(
    "use_known", [True, False], ids=["with_claim_form_values", "detection_only"]
)
def test_pii_is_masked_and_required_values_survive(use_known: bool, known) -> None:
    masker = PIIMasker(known if use_known else {})

    masked = masker.mask(SAMPLE)

    for value in PII_VALUES:
        assert value not in masked, f"{value!r} leaked"
    for value in MUST_SURVIVE:
        assert value in masked, f"{value!r} was masked but extraction needs it"
    for prefix in ("<PERSON_", "<PHONE_", "<EMAIL_", "<AADHAAR_", "<PAN_", "<ADDRESS_"):
        assert prefix in masked


def test_unmask_restores_the_original_text_exactly(known) -> None:
    masker = PIIMasker(known)
    assert masker.unmask(masker.mask(SAMPLE)) == SAMPLE


def test_placeholders_are_stable_and_values_are_restored_individually(known) -> None:
    masker = PIIMasker(known)

    masked = masker.mask(
        "Suhani Bala called 07649716752. Later SUHANI   BALA called 07649716752 again."
    )

    assert masked.count("<PERSON_1>") == 2  # same person (any case/spacing) = same placeholder
    assert masked.count("<PHONE_1>") == 2
    assert masker.unmask_value("<PERSON_1>") == "Suhani Bala"
    assert (
        masker.unmask_value("name: <person_1>, tel <PHONE_1>")
        == "name: Suhani Bala, tel 07649716752"
    )
    assert masker.unmask_value("<PERSON_9> is unknown") == "<PERSON_9> is unknown"
    assert masker.unmask_value(None) is None


def test_the_mapping_is_never_printed(known) -> None:
    masker = PIIMasker(known)
    masker.mask(SAMPLE)
    assert "Suhani" not in repr(masker) and "07649716752" not in repr(masker) and masker.count > 3
    assert "Suhani" not in str(masker)


def test_amounts_dates_and_codes_are_not_mistaken_for_pii() -> None:
    text = "Total 1,50,000 Rs. 77917 on 14-11-2025 ICD O80 policy POL-AS/2024/1234567 bill 22349"
    assert PIIMasker().mask(text) == text


def test_aadhaar_like_numbers_and_pan_are_masked_without_claim_form_help() -> None:
    masker = PIIMasker()
    masked = masker.mask("Aadhaar: 4567 1234 9876 and PAN CDEPK4321L, mobile +91 98765 43210")
    assert "4567 1234 9876" not in masked and "CDEPK4321L" not in masked
    assert "98765 43210" not in masked.replace("+91 ", "") or "<PHONE_" in masked


def test_logs_contain_no_raw_phone_or_email(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logger = logging.getLogger("intake.test")

    logger.info("call from 07649716752 or mail suhani15@example.com aadhaar 2345 6789 0123")
    logger.info(
        "structured", extra={"phone": "+91 76497 16752", "nested": {"mail": "a.b@x.in", "ok": 5}}
    )
    try:
        raise ValueError("failed for 9876543210")
    except ValueError:
        logger.exception("boom")

    out = capsys.readouterr().out
    for raw in (
        "07649716752",
        "suhani15@example.com",
        "2345 6789 0123",
        "a.b@x.in",
        "9876543210",
        "76497 16752",
    ):
        assert raw not in out, f"{raw!r} reached the log output"
    assert "[phone]" in out and "[email]" in out and "[aadhaar]" in out
    lines = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert any(line["message"] == "structured" and line["nested"]["ok"] == 5 for line in lines)


def test_the_filter_works_on_any_handler() -> None:
    record = logging.LogRecord("x", logging.INFO, "f", 1, "mail %s", ("me@mail.com",), None)
    PIIRedactingFilter().filter(record)
    assert record.getMessage() == "mail [email]"
    assert (
        redact("keep amount 77917 and policy POL-A/2025/1234567")
        == "keep amount 77917 and policy POL-A/2025/1234567"
    )
