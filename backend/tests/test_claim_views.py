"""The read endpoints behind the web UI: claims list and detail, document files, a clause by id, and
the hints a reload of the claim page needs (latest run, event timestamps)."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from agent_helpers import make_agent_world
from auth_helpers import bearer
from fake_redis import FakeRedis
from sqlalchemy.ext.asyncio import AsyncSession
from test_auth import client_for, make_settings, make_user

from app.api.claim_views import mask_phone
from app.api.claims import get_intake_service
from app.intake.storage import LocalStorage
from app.models import (
    ApprovalAction,
    Claim,
    ClaimApproval,
    ClaimDecision,
    ClaimDocument,
    ClaimStatus,
    DecisionType,
    DocType,
    UserRole,
)

PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"


class StubIntake:
    def __init__(self, root: Path) -> None:
        self.storage = LocalStorage(root)


async def add_document(
    session: AsyncSession, claim: Claim, root: Path, name: str = "bill.pdf", data: bytes = PDF
) -> ClaimDocument:
    key = f"claims/{claim.claim_number}/{name}"
    (root / key).parent.mkdir(parents=True, exist_ok=True)
    (root / key).write_bytes(data)
    document = ClaimDocument(
        claim_id=claim.id,
        doc_type=DocType.HOSPITAL_BILL,
        file_path=key,
        mime_type="application/pdf",
        page_count=2,
        sha256="a" * 64,
    )
    session.add(document)
    await session.flush()
    return document


def test_phone_numbers_keep_only_the_last_four_digits() -> None:
    assert mask_phone("9876543210") == "******3210"
    assert mask_phone("+91 98765 43210") == "********3210"
    assert mask_phone("123") == "***" and mask_phone(None) is None and mask_phone("") is None


async def test_list_filters_searches_and_pages(session: AsyncSession) -> None:
    officer = await make_user(session, "off@example.com")
    world = await make_agent_world(session)
    base = world.claim
    extra = []
    for i, status in enumerate((ClaimStatus.APPROVED, ClaimStatus.REJECTED, ClaimStatus.SUBMITTED)):
        claim = Claim(
            claim_number=f"LIST-{i:03d}",
            customer_policy_id=base.customer_policy_id,
            claimed_amount=Decimal("1000") * (i + 1),
            status=status,
            hospital_name=f"Hospital {i}",
        )
        session.add(claim)
        extra.append(claim)
    await session.flush()
    for i, claim in enumerate(extra):  # a known order: the last one is the newest
        claim.updated_at = datetime.now(UTC) - timedelta(minutes=10 - i)
    await session.flush()
    settings = make_settings()
    async with client_for(session, settings) as (http, _, _):
        h = bearer(settings, officer)
        everything = (await http.get("/claims", headers=h, params={"size": 100})).json()
        assert everything["total"] == 4 and everything["page"] == 1
        first = everything["items"][0]
        assert (
            first["customer_name"] == "Asha Verma"
            and first["customer_phone_masked"] == "******3210"
        )
        assert "9876543210" not in str(everything)  # the full number never leaves
        assert first["policy_code"] == "POL-T"
        approved = (await http.get("/claims", headers=h, params={"status": "approved"})).json()
        assert [c["claim_number"] for c in approved["items"]] == ["LIST-000"]
        two = (
            await http.get(
                "/claims", headers=h, params=[("status", "approved"), ("status", "rejected")]
            )
        ).json()
        assert {c["claim_number"] for c in two["items"]} == {"LIST-000", "LIST-001"}
        found = (await http.get("/claims", headers=h, params={"q": "list-00"})).json()
        assert found["total"] == 3
        literal = (await http.get("/claims", headers=h, params={"q": "%"})).json()
        assert literal["total"] == 0  # a wildcard is a character, not a pattern
        paged = (await http.get("/claims", headers=h, params={"size": 2, "page": 2})).json()
        assert len(paged["items"]) == 2 and paged["total"] == 4
        assert (
            await http.get("/claims", headers=h, params={"status": "nonsense"})
        ).status_code == 422
        assert (await http.get("/claims", headers=h, params={"size": 500})).status_code == 422
        assert (await http.get("/claims")).status_code == 401


async def test_detail_has_documents_history_and_separation_of_duties_hint(
    session: AsyncSession, tmp_path: Path
) -> None:
    uploader = await make_user(session, "uploader@example.com")
    other = await make_user(session, "other@example.com")
    admin = await make_user(session, "boss@example.com", UserRole.ADMIN)
    world = await make_agent_world(session)
    claim = world.claim
    claim.created_by = uploader.id
    await add_document(session, claim, tmp_path)
    older = ClaimDecision(
        claim_id=claim.id,
        decision=DecisionType.FLAG_FOR_REVIEW,
        payable_amount=Decimal("0"),
        agent_run_id="run-old",
        model_used="gpt-4o-mini",
    )
    newer = ClaimDecision(
        claim_id=claim.id,
        decision=DecisionType.PARTIAL_APPROVE,
        payable_amount=Decimal("53666.86"),
        agent_run_id="run-new",
        model_used="claude-haiku-5-5",
    )
    session.add_all([older, newer])
    await session.flush()
    older.created_at = datetime.now(UTC) - timedelta(days=1)
    session.add(
        ClaimApproval(
            claim_id=claim.id,
            decision_id=older.id,
            reviewer_id=admin.id,
            action=ApprovalAction.REJECTED,
            comment="duplicate claim",
        )
    )
    await session.flush()
    settings = make_settings()
    redis = FakeRedis()
    redis.values[f"claim:{claim.id}:latest_run"] = "run-live"
    async with client_for(session, settings, redis=redis) as (http, _, _):
        mine = (
            await http.get(f"/claims/{claim.claim_number}", headers=bearer(settings, uploader))
        ).json()
        theirs = (
            await http.get(f"/claims/{claim.claim_number}", headers=bearer(settings, other))
        ).json()
        assert mine["uploaded_by_me"] is True and theirs["uploaded_by_me"] is False
        assert mine["claim_number"] == claim.claim_number and mine["policy_code"] == "POL-T"
        assert mine["customer"] == {"name": "Asha Verma", "phone_masked": "******3210"}
        assert [d["file_name"] for d in mine["documents"]] == ["bill.pdf"]
        assert mine["documents"][0]["doc_type"] == "hospital_bill"
        assert mine["latest_run_id"] == "run-live"  # a run started after the last decision
        runs = [h["run_id"] for h in mine["history"]]
        assert runs == ["run-new", "run-old"]  # newest first
        assert mine["history"][0]["approval"] is None
        approval = mine["history"][1]["approval"]
        assert approval["action"] == "rejected" and approval["reviewer_name"] == "Boss"
        assert approval["comment"] == "duplicate claim"
        assert "9876543210" not in str(mine)
        assert (
            await http.get("/claims/NOPE-1", headers=bearer(settings, other))
        ).status_code == 404

    # without the Redis hint the latest decision's run is the best guess
    async with client_for(session, settings) as (http, _, _):
        detail = (
            await http.get(f"/claims/{claim.claim_number}", headers=bearer(settings, other))
        ).json()
    assert detail["latest_run_id"] == "run-new"


async def test_document_file_is_served_inline_from_storage(
    session: AsyncSession, tmp_path: Path
) -> None:
    officer = await make_user(session, "off@example.com")
    world = await make_agent_world(session)
    document = await add_document(session, world.claim, tmp_path)
    other_claim = Claim(claim_number="OTHER-1", claimed_amount=Decimal("1"))
    session.add(other_claim)
    await session.flush()
    settings = make_settings()
    async with client_for(session, settings) as (http, _, _):
        http._transport.app.dependency_overrides[get_intake_service] = lambda: StubIntake(tmp_path)  # type: ignore[attr-defined]
        h = bearer(settings, officer)
        ok = await http.get(f"/claims/{world.claim.id}/documents/{document.id}/file", headers=h)
        assert ok.status_code == 200 and ok.content == PDF
        assert ok.headers["content-type"] == "application/pdf"
        assert ok.headers["content-disposition"] == 'inline; filename="bill.pdf"'
        wrong_claim = await http.get(
            f"/claims/{other_claim.id}/documents/{document.id}/file", headers=h
        )
        assert wrong_claim.status_code == 404  # a document is only reachable through its own claim
        (tmp_path / document.file_path).unlink()
        assert (
            await http.get(f"/claims/{world.claim.id}/documents/{document.id}/file", headers=h)
        ).status_code == 404
        assert (
            await http.get(f"/claims/{world.claim.id}/documents/{document.id}/file")
        ).status_code == 401


async def test_a_clause_can_be_read_by_id(session: AsyncSession) -> None:
    officer = await make_user(session, "off@example.com")
    world = await make_agent_world(session)
    settings = make_settings()
    async with client_for(session, settings) as (http, _, _):
        h = bearer(settings, officer)
        resp = await http.get(f"/policies/chunks/{world.chunk.id}", headers=h)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["policy_code"] == "POL-T" and body["content"] == world.chunk.content
        assert body["section_path"] == world.chunk.section_path and body["clauses"] == ["6.3"]
        assert (body["page_start"], body["page_end"]) == (8, 9)
        assert (
            await http.get("/policies/chunks/00000000-0000-0000-0000-000000000001", headers=h)
        ).status_code == 404
        assert (await http.get("/policies/chunks/not-a-uuid", headers=h)).status_code == 422
        assert (await http.get(f"/policies/chunks/{world.chunk.id}")).status_code == 401


async def test_starting_a_run_remembers_it_for_a_reload(session: AsyncSession) -> None:
    officer = await make_user(session, "off@example.com")
    world = await make_agent_world(session)
    settings = make_settings()
    redis = FakeRedis()
    async with client_for(session, settings, redis=redis) as (http, _, _):
        resp = await http.post(
            f"/claims/{world.claim.id}/process", headers=bearer(settings, officer)
        )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]
    assert redis.values[f"claim:{world.claim.id}:latest_run"] == run_id
    assert 0 < redis.expiry[f"claim:{world.claim.id}:latest_run"] - redis.now <= 7 * 86400
