"""IntakeService + /claims API on real generated claims (data/claims) with a mocked LLM and OCR."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from auth_helpers import FAKE_ADMIN, bypass_auth
from fastapi.testclient import TestClient
from intake_fakes import (
    BILL_TEXT,
    SUMMARY_TEXT,
    FakeOCR,
    ScriptedChat,
    good_llm_extraction,
    intake_settings,
    make_pdf,
    placeholder_after,
)
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.intake.schemas import LLMExtraction, LLMLineItem, LLMObservation
from app.intake.service import IntakeService
from app.main import create_app
from app.models import (
    AuditLog,
    Claim,
    ClaimDocument,
    ClaimSource,
    ClaimStatus,
    Customer,
    CustomerPolicy,
    DocType,
    Policy,
)
from app.repositories.extraction import ExtractionRepository

DATA = get_settings().data_dir
TRUTH_PATH = DATA / "golden" / "extraction_truth.json"
pytestmark = pytest.mark.skipif(
    not (TRUTH_PATH.exists() and (DATA / "claims" / "CLM-001").is_dir()),
    reason="generated claims / extraction_truth.json not available",
)


def truth_for(claim_id: str) -> dict:
    return next(r for r in json.loads(TRUTH_PATH.read_text("utf-8")) if r["claim_id"] == claim_id)


def llm_from_truth(claim_id: str):
    """What a perfect model would answer for this claim (built from the generator's truth)."""
    t = truth_for(claim_id)
    f = t["fields"]

    def answer(messages) -> LLMExtraction:
        name = placeholder_after("Patient Name", messages) or f["patient_name"]
        return LLMExtraction(
            patient_name=name,
            hospital_name=f["hospital_name"],
            admission_date=f["admission_date"],
            discharge_date=f["discharge_date"],
            diagnosis_text=f["diagnosis_text"],
            icd10_code=f["icd10_code"],
            line_items=[
                LLMLineItem(description=i["description"], category="other", quantity=i["qty"],
                            unit_price=i["rate"], amount=i["amount"])
                for i in t["line_items"]
            ],
            bill_subtotal=f["bill_subtotal"],
            taxes=f["taxes"],
            bill_total=f["bill_total"],
            observations=[
                LLMObservation(doc_type="hospital_bill", patient_name=name,
                               admission_date=f["admission_date"], discharge_date=f["discharge_date"],
                               total_amount=f["bill_total"]),
                *(
                    [LLMObservation(doc_type="discharge_summary", patient_name=name,
                                    admission_date=t["documents"]["summary_admission_date"],
                                    discharge_date=t["documents"]["summary_discharge_date"])]
                    if t["documents"]["discharge_summary_present"] else []
                ),
            ],
        )  # fmt: skip

    return answer


async def add_seeded_claim(factory, claim_id: str) -> Claim:
    """Insert a claim + its documents the way `make seed` does (files stay in data/claims)."""
    form = json.loads((DATA / "claims" / claim_id / "claim_form.json").read_text("utf-8"))
    async with factory() as session:
        policy = (
            await session.execute(select(Policy).where(Policy.policy_code == form["policy_id"]))
        ).scalar_one_or_none()
        if policy is None:
            policy = Policy(policy_code=form["policy_id"], name=form["policy_id"], rules={})
            session.add(policy)
            await session.flush()
        customer = Customer(
            full_name=form["claimant"]["name"], phone=form["claimant"]["phone"],
            email=form["claimant"]["email"], address=form["claimant"]["address"],
        )  # fmt: skip
        session.add(customer)
        await session.flush()
        link = None
        if form["policy_number"]:
            link = CustomerPolicy(customer_id=customer.id, policy_id=policy.id,
                                  policy_number=form["policy_number"], sum_insured=Decimal(form["sum_insured"]),
                                  start_date=date.fromisoformat(form["policy_start_date"]))  # fmt: skip
            session.add(link)
            await session.flush()
        claim = Claim(claim_number=claim_id, customer_policy_id=link.id if link else None,
                      claimed_amount=Decimal(form["claimed_amount"]), source=ClaimSource.SEED)  # fmt: skip
        session.add(claim)
        await session.flush()
        for path in sorted((DATA / "claims" / claim_id).iterdir()):
            doc_type = (
                DocType.CLAIM_FORM if path.name == "claim_form.json"
                else DocType.HOSPITAL_BILL if path.name.startswith("hospital_bill")
                else DocType.DISCHARGE_SUMMARY
            )  # fmt: skip
            session.add(ClaimDocument(
                claim_id=claim.id, doc_type=doc_type, file_path=f"claims/{claim_id}/{path.name}",
                mime_type="application/octet-stream", is_scanned="scanned" in path.name,
                page_count=1, sha256=path.name.ljust(64, "0"),
            ))  # fmt: skip
        await session.commit()
        return claim


@pytest.fixture
async def factory(test_database_url: str):
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    yield session_factory
    async with session_factory() as session:
        await session.execute(
            text(
                "TRUNCATE claim_documents, claim_extractions, claims, customer_policies, customers, "
                "policies, audit_logs CASCADE"
            )
        )
        await session.commit()
    await engine.dispose()


def service_for(factory, chat, tmp_path: Path) -> IntakeService:
    return IntakeService(factory, chat, ocr=FakeOCR(), settings=intake_settings(storage_dir=tmp_path),
                         data_dir=DATA)  # fmt: skip


# ---------------------------------------------------------------- process a seeded claim


async def test_a_seeded_claim_is_extracted_end_to_end(factory, tmp_path) -> None:
    claim = await add_seeded_claim(factory, "CLM-001")
    t = truth_for("CLM-001")
    chat = ScriptedChat([llm_from_truth("CLM-001")])
    service = service_for(factory, chat, tmp_path)

    result = await service.process(claim.id)

    assert result.status == "completed" and result.version == 1 and result.missing_documents == []
    assert result.validation_errors == [] and result.claim_status == "processing"
    assert set(result.timings_ms) >= {
        "text_extraction",
        "classify",
        "extraction",
        "consistency",
        "total",
    }

    async with factory() as session:
        row = await ExtractionRepository(session).latest_for_claim(claim.id)
        stored = (await session.execute(select(Claim).where(Claim.id == claim.id))).scalar_one()
        actions = [
            a.action
            for a in (
                await session.execute(select(AuditLog).order_by(AuditLog.created_at))
            ).scalars()
        ]
    data = row.extracted_data
    f = t["fields"]
    assert data["patient_name"] == f["patient_name"]  # restored after masking
    assert (data["admission_date"], data["discharge_date"]) == (
        f["admission_date"],
        f["discharge_date"],
    )
    assert (
        Decimal(data["bill_total"]) == Decimal(f["bill_total"])
        and len(data["line_items"]) == f["line_item_count"]
    )
    assert Decimal(data["claimed_amount"]) == Decimal(
        f["claimed_amount"]
    )  # from the JSON claim form
    assert data["field_sources"]["claimed_amount"]["method"] == "structured_json"
    assert data["field_confidence"]["icd10_code"] == "high" and data["missing_documents"] == []
    assert data["intake"]["pii_placeholders"] >= 3 and row.model_used == "fake-model"
    # the claim columns that were empty are now filled; status and audit trail are written
    assert stored.icd10_code == f["icd10_code"] and stored.diagnosis_text == f["diagnosis_text"]
    assert "claim.status_changed" in actions and "intake.extracted" in actions

    prompt = chat.all_prompt_text()
    for raw in (t["pii"]["name"], t["pii"]["phone"], t["pii"]["email"], t["pii"]["address"]):
        assert raw not in prompt, f"{raw!r} reached the LLM"

    again = await service.process(claim.id)  # a second run adds a NEW version
    assert again.version == 2


async def test_digital_copy_is_preferred_and_the_scan_is_set_aside(factory, tmp_path) -> None:
    scanned = next(
        r["claim_id"] for r in json.loads(TRUTH_PATH.read_text("utf-8"))
        if r["has_scanned_copy"] and r["documents"]["discharge_summary_present"]
    )  # fmt: skip
    claim = await add_seeded_claim(factory, scanned)
    ocr = FakeOCR(BILL_TEXT)
    service = IntakeService(factory, ScriptedChat([llm_from_truth(scanned)]), ocr=ocr,
                            settings=intake_settings(storage_dir=tmp_path), data_dir=DATA)  # fmt: skip

    result = await service.process(claim.id, prefer="digital")

    async with factory() as session:
        data = (await ExtractionRepository(session).latest_for_claim(claim.id)).extracted_data
    used = {d["file"]: d["used"] for d in data["intake"]["documents"]}
    assert used["hospital_bill.pdf"] is True and used["hospital_bill_scanned.png"] is False
    assert any(i["type"] == "document_superseded" for i in data["consistency_issues"])
    assert result.status == "completed" and ocr.calls >= 1  # the scan WAS read, then set aside


async def test_scanned_mode_extracts_from_the_ocr_copy(factory, tmp_path) -> None:
    scanned = next(
        r["claim_id"] for r in json.loads(TRUTH_PATH.read_text("utf-8")) if r["has_scanned_copy"]
    )
    claim = await add_seeded_claim(factory, scanned)
    service = IntakeService(factory, ScriptedChat([llm_from_truth(scanned)]), ocr=FakeOCR(BILL_TEXT, 90.0),
                            settings=intake_settings(storage_dir=tmp_path), data_dir=DATA)  # fmt: skip

    await service.process(claim.id, prefer="scanned")

    async with factory() as session:
        data = (await ExtractionRepository(session).latest_for_claim(claim.id)).extracted_data
    used = {d["file"]: d["used"] for d in data["intake"]["documents"]}
    assert used["hospital_bill_scanned.png"] is True and used["hospital_bill.pdf"] is False
    assert data["intake"]["prefer"] == "scanned"


async def test_missing_discharge_summary_sets_documents_requested(factory, tmp_path) -> None:
    missing_id = next(
        r["claim_id"] for r in json.loads(TRUTH_PATH.read_text("utf-8"))
        if r["expected_missing_documents"] == ["discharge_summary"]
    )  # fmt: skip
    claim = await add_seeded_claim(factory, missing_id)
    service = service_for(factory, ScriptedChat([llm_from_truth(missing_id)]), tmp_path)

    result = await service.process(claim.id)

    assert result.missing_documents == ["discharge_summary"]
    assert result.claim_status == ClaimStatus.DOCUMENTS_REQUESTED.value
    assert any(i.type == "missing_document" and i.severity == "critical" for i in result.issues)


async def test_a_failing_llm_marks_the_claim_failed_and_is_recorded(factory, tmp_path) -> None:
    claim = await add_seeded_claim(factory, "CLM-001")

    class Down(ScriptedChat):
        def with_structured_output(self, *a, **k):
            raise RuntimeError("provider down")

    result = await service_for(factory, Down(), tmp_path).process(claim.id)

    assert result.status == "failed" and "provider down" in result.error
    async with factory() as session:
        stored = (await session.execute(select(Claim).where(Claim.id == claim.id))).scalar_one()
        assert stored.status == ClaimStatus.FAILED
        assert await ExtractionRepository(session).latest_for_claim(claim.id) is None


# ---------------------------------------------------------------- upload


async def test_duplicate_upload_of_the_same_file_is_ignored(factory, tmp_path) -> None:
    service = service_for(factory, ScriptedChat([good_llm_extraction()]), tmp_path)
    pdf = (DATA / "claims" / "CLM-002" / "hospital_bill.pdf").read_bytes()

    async with factory() as session:
        claim, ignored_first = await service.create_claim(
            session, policy_number="POL-X/2025/1", files=[("a.pdf", pdf), ("copy-of-a.pdf", pdf)]
        )
        await session.commit()
        assert ignored_first == ["copy-of-a.pdf"]  # same bytes twice in one request
        created, ignored = await service.add_documents(session, claim, [("again.pdf", pdf)])
        await session.commit()
        assert created == [] and ignored == ["again.pdf"]  # and again in a later request
        rows = (
            (await session.execute(select(ClaimDocument).where(ClaimDocument.claim_id == claim.id)))
            .scalars()
            .all()
        )

    assert len(rows) == 1
    assert rows[0].file_path == f"claims/{claim.claim_number}/{rows[0].sha256}.pdf"
    assert (tmp_path / rows[0].file_path).read_bytes() == pdf  # stored once, under its hash
    assert claim.source == ClaimSource.UPLOAD and claim.claim_number.startswith("UP-")


async def test_a_bad_file_rejects_the_whole_upload_and_leaves_no_claim(factory, tmp_path) -> None:
    from app.intake.storage import UploadError

    service = service_for(factory, ScriptedChat([good_llm_extraction()]), tmp_path)
    good = make_pdf([BILL_TEXT])

    async with factory() as session:
        with pytest.raises(UploadError, match="not a PDF, PNG or JPEG"):
            await service.create_claim(
                session,
                policy_number="POL-X/1",
                files=[("ok.pdf", good), ("evil.pdf", b"MZ\x90\x00" + b"0" * 100)],
            )
        await session.rollback()
        assert (await session.execute(select(Claim))).scalars().all() == []
    assert not (tmp_path / "claims").exists() or not list((tmp_path / "claims").rglob("*.pdf"))


# ---------------------------------------------------------------- HTTP API


@pytest.fixture
def api(test_database_url: str, tmp_path):
    settings = Settings(_env_file=None, app_env="test", database_url=test_database_url)
    app = create_app(settings)
    bypass_auth(app)
    chat = ScriptedChat(
        [
            lambda m: good_llm_extraction(
                patient_name=placeholder_after("Patient Name", m) or "<PERSON_1>"
            )
        ]
    )
    import asyncio

    async def seed_acting_user() -> None:  # claims.created_by points at a real user
        engine = create_async_engine(test_database_url, poolclass=NullPool)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO users (id, email, full_name, role, password_hash) "
                    "VALUES (:id, :email, 'Test Admin', 'admin', 'x') ON CONFLICT DO NOTHING"
                ),
                {"id": str(FAKE_ADMIN.id), "email": FAKE_ADMIN.email},
            )
        await engine.dispose()

    asyncio.run(seed_acting_user())
    with TestClient(app) as client:
        app.state.intake_service = IntakeService(
            app.state.session_factory, chat, ocr=FakeOCR(),
            settings=intake_settings(storage_dir=tmp_path), data_dir=DATA,
        )  # fmt: skip
        client.chat = chat  # type: ignore[attr-defined]
        yield client
    import asyncio

    async def cleanup() -> None:
        engine = create_async_engine(test_database_url, poolclass=NullPool)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "TRUNCATE claim_documents, claim_extractions, claims, customer_policies, "
                    "customers, policies, audit_logs, users CASCADE"
                )
            )
        await engine.dispose()

    asyncio.run(cleanup())


def test_upload_rejects_a_renamed_exe_over_http(api: TestClient) -> None:
    response = api.post("/claims", data={"policy_number": "POL-X/2025/1"},
                        files=[("files", ("bill.pdf", b"MZ\x90\x00\x03" + b"\x00" * 300, "application/pdf"))])  # fmt: skip

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_upload"
    assert "not a PDF, PNG or JPEG" in response.json()["error"]["message"]


def test_upload_limits_over_http(api: TestClient) -> None:
    too_many = [
        ("files", (f"f{i}.pdf", make_pdf([f"page {i} " * 20]), "application/pdf"))
        for i in range(11)
    ]
    assert api.post("/claims", data={"policy_number": "POL-X/1"}, files=too_many).status_code == 400
    assert api.post("/claims", data={"policy_number": "POL-X/1"}).status_code == 422  # no files


def test_upload_then_extract_then_read_the_extraction(api: TestClient) -> None:
    bill = make_pdf([BILL_TEXT])  # one bytes object: re-making a PDF gives different bytes
    files = [
        ("files", ("bill.pdf", bill, "application/pdf")),
        ("files", ("summary.pdf", make_pdf([SUMMARY_TEXT]), "application/pdf")),
        ("files", ("bill-again.pdf", bill, "application/pdf")),  # identical bytes: ignored
    ]

    created = api.post(
        "/claims", data={"policy_number": "POL-X/2025/9", "claim_number": "UP-TEST-1"}, files=files
    )
    assert created.status_code == 201
    body = created.json()
    assert body["claim_number"] == "UP-TEST-1" and body["status"] == "submitted"
    assert (
        len(body["documents"]) == 2 and body["duplicates_ignored"] == 1
    )  # identical bytes ignored
    assert (
        api.post(
            "/claims",
            data={"policy_number": "POL-X/2025/9", "claim_number": "UP-TEST-1"},
            files=files[:1],
        ).status_code
        == 409
    )

    early = api.get(f"/claims/{body['id']}/extraction")
    assert early.status_code == 404 and "no extraction yet" in early.json()["error"]["message"]

    started = api.post(f"/claims/{body['id']}/extract")
    assert started.status_code == 202 and started.json()["status"] == "processing"

    result = api.get(
        f"/claims/{body['id']}/extraction"
    )  # background task finished inside the client
    assert result.status_code == 200
    data = result.json()
    assert (
        data["version"] == 1
        and data["claim_status"] == "processing"
        and data["missing_documents"] == []
    )
    assert data["extraction"]["patient_name"] == "Suhani Bala"
    assert data["extraction"]["icd10_code"] == "O80" and data["model_used"] == "fake-model"
    kinds = {
        d["doc_type"]
        for d in api.post(
            f"/claims/{body['id']}/documents",
            files=[("files", ("x.pdf", make_pdf([BILL_TEXT]), "application/pdf"))],
        ).json()["documents"]
    }
    assert kinds == {"hospital_bill", "discharge_summary"}  # classified by rules during extraction
    assert api.get("/claims/00000000-0000-0000-0000-000000000000/extraction").status_code == 404
    assert api.post("/claims/00000000-0000-0000-0000-000000000000/extract").status_code == 404
