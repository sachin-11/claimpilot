from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models import DocType
from app.repositories import ClaimRepository
from app.scripts.seed import classify_document, run_seed


@pytest.fixture
def seed_settings() -> Settings:
    settings = Settings(
        _env_file=None, seed_admin_password="admin-pw", seed_officer_password="officer-pw"
    )
    if not (settings.data_dir / "claims").is_dir() or not settings.policy_rules_path.is_file():
        pytest.skip("generated claims / policy_rules.yaml not available")
    return settings


async def test_seed_is_idempotent(session: AsyncSession, seed_settings: Settings) -> None:
    first = await run_seed(session, seed_settings)
    second = await run_seed(session, seed_settings)

    assert first == second
    assert first["policies"] >= 1
    assert first["claims"] == len(
        list((seed_settings.data_dir / "claims").glob("*/claim_form.json"))
    )
    assert first["users"] == 2
    assert first["claim_documents"] >= first["claims"]


async def test_seeded_claim_has_documents_and_no_golden_data(
    session: AsyncSession, seed_settings: Settings
) -> None:
    await run_seed(session, seed_settings)

    claim = await ClaimRepository(session).get_by_number("CLM-001")
    assert claim is not None
    assert claim.status.value == "submitted" and claim.source.value == "seed"
    assert claim.icd10_code is None  # only present in the golden data, which is never loaded
    assert {d.doc_type for d in claim.documents} >= {DocType.CLAIM_FORM, DocType.HOSPITAL_BILL}


def test_classify_document() -> None:
    assert classify_document(Path("hospital_bill.pdf")) == (DocType.HOSPITAL_BILL, False)
    assert classify_document(Path("discharge_summary_scanned.png")) == (
        DocType.DISCHARGE_SUMMARY,
        True,
    )
    assert classify_document(Path("claim_form.json")) == (DocType.CLAIM_FORM, False)
    assert classify_document(Path("notes.txt")) == (DocType.OTHER, False)
