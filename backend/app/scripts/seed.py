"""Seed the database: policies, customers, claims + documents, and two users.

    python -m app.scripts.seed

Idempotent: every row is looked up by a natural key first, so a second run changes nothing.
data/golden is deliberately never read here. Expected answers are for evaluation only.
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import bcrypt
import yaml
from pypdf import PdfReader
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.db import Base
from app.core.db import create_engine as create_db_engine
from app.core.logging import setup_logging
from app.models import (
    Claim,
    ClaimDocument,
    ClaimSource,
    Customer,
    CustomerPolicy,
    DocType,
    Policy,
    User,
    UserRole,
)
from app.repositories import ClaimRepository, PolicyRepository
from app.schemas.claim import ClaimCreate

logger = logging.getLogger("seed")

SUMMARY_TABLES = [
    "policies",
    "customers",
    "customer_policies",
    "claims",
    "claim_documents",
    "users",
    "audit_logs",
]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def classify_document(path: Path) -> tuple[DocType, bool]:
    """(doc_type, is_scanned) from a file name like hospital_bill_scanned.png."""
    stem = path.stem
    scanned = stem.endswith("_scanned")
    if stem.startswith("hospital_bill"):
        return DocType.HOSPITAL_BILL, scanned
    if stem.startswith("discharge_summary"):
        return DocType.DISCHARGE_SUMMARY, scanned
    if stem == "claim_form":
        return DocType.CLAIM_FORM, False
    return DocType.OTHER, scanned


def page_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return len(PdfReader(path).pages)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return 1
    return None


def load_policy_rules(rules_path: Path) -> dict[str, Any]:
    return yaml.safe_load(rules_path.read_text(encoding="utf-8"))


async def seed_policies(session: AsyncSession, rules_path: Path) -> dict[str, Policy]:
    data = await asyncio.to_thread(load_policy_rules, rules_path)
    repo = PolicyRepository(session)
    policies: dict[str, Policy] = {}
    for entry in data["policies"]:
        policy, _ = await repo.upsert_from_rules(entry)
        policies[policy.policy_code] = policy
    return policies


async def _get_or_create_customer(session: AsyncSession, claimant: dict[str, Any]) -> Customer:
    # The claim form has no customer id, so name + phone identify a person.
    result = await session.execute(
        select(Customer).where(
            Customer.full_name == claimant["name"], Customer.phone == claimant.get("phone")
        )
    )
    customer = result.scalars().first()
    if customer is None:
        customer = Customer(
            full_name=claimant["name"],
            phone=claimant.get("phone"),
            email=claimant.get("email"),
            address=claimant.get("address"),
        )
        session.add(customer)
        await session.flush()
    return customer


async def _get_or_create_customer_policy(
    session: AsyncSession, form: dict[str, Any], customer: Customer, policy: Policy
) -> CustomerPolicy:
    result = await session.execute(
        select(CustomerPolicy).where(CustomerPolicy.policy_number == form["policy_number"])
    )
    customer_policy = result.scalar_one_or_none()
    if customer_policy is None:
        customer_policy = CustomerPolicy(
            customer_id=customer.id,
            policy_id=policy.id,
            policy_number=form["policy_number"],
            sum_insured=Decimal(form["sum_insured"]),
            start_date=date.fromisoformat(form["policy_start_date"]),
        )
        session.add(customer_policy)
        await session.flush()
    return customer_policy


def scan_documents(claim_dir: Path, data_dir: Path) -> list[ClaimDocument]:
    """One (unattached) ClaimDocument per file in the claim folder."""
    documents = []
    for path in sorted(claim_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        doc_type, scanned = classify_document(path)
        documents.append(
            ClaimDocument(
                doc_type=doc_type,
                file_path=path.relative_to(data_dir).as_posix(),
                mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                is_scanned=scanned,
                page_count=page_count(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return documents


async def _sync_documents(
    session: AsyncSession, claim: Claim, claim_dir: Path, data_dir: Path
) -> None:
    result = await session.execute(
        select(ClaimDocument.file_path).where(ClaimDocument.claim_id == claim.id)
    )
    known = set(result.scalars())
    for document in await asyncio.to_thread(scan_documents, claim_dir, data_dir):
        if document.file_path not in known:
            document.claim_id = claim.id
            session.add(document)
    await session.flush()


async def seed_claims(session: AsyncSession, data_dir: Path, policies: dict[str, Policy]) -> None:
    claims = ClaimRepository(session)
    form_paths = await asyncio.to_thread(
        lambda: sorted((data_dir / "claims").glob("*/claim_form.json"))
    )
    for form_path in form_paths:
        form = await asyncio.to_thread(
            lambda p=form_path: json.loads(p.read_text(encoding="utf-8"))
        )
        policy = policies.get(form["policy_id"])
        if policy is None:
            logger.warning("seed_unknown_policy", extra={"claim": form["claim_id"]})
            continue

        claim = await claims.get_by_number(form["claim_id"])
        if claim is None:
            customer = await _get_or_create_customer(session, form["claimant"])
            customer_policy = None
            if form.get("policy_number"):  # some claims deliberately lack one
                customer_policy = await _get_or_create_customer_policy(
                    session, form, customer, policy
                )
            claim = await claims.create(
                ClaimCreate(
                    claim_number=form["claim_id"],
                    customer_policy_id=customer_policy.id if customer_policy else None,
                    hospital_name=form.get("hospital_name"),
                    diagnosis_text=form.get("diagnosis"),
                    admission_date=form.get("admission_date"),
                    discharge_date=form.get("discharge_date"),
                    claimed_amount=Decimal(form["claimed_amount"]),
                    source=ClaimSource.SEED,
                ),
            )
        await _sync_documents(session, claim, form_path.parent, data_dir)


async def seed_users(session: AsyncSession, settings: Settings) -> None:
    wanted = [
        (settings.seed_admin_email, "Seed Admin", UserRole.ADMIN, settings.seed_admin_password),
        (
            settings.seed_officer_email,
            "Seed Officer",
            UserRole.OFFICER,
            settings.seed_officer_password,
        ),
    ]
    for email, full_name, role, password in wanted:
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none() is not None:
            continue
        if password is None or not password.get_secret_value():
            raise SystemExit(
                f"Set SEED_{role.name}_PASSWORD in .env to create the {role.value} user."
            )
        session.add(
            User(
                email=email,
                full_name=full_name,
                role=role,
                password_hash=hash_password(password.get_secret_value()),
            )
        )
    await session.flush()


async def count_rows(session: AsyncSession) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in SUMMARY_TABLES:
        table = Base.metadata.tables[name]
        counts[name] = (await session.execute(select(func.count()).select_from(table))).scalar_one()
    return counts


async def run_seed(session: AsyncSession, settings: Settings) -> dict[str, int]:
    """Seed everything inside the given session (the caller commits). Returns row counts."""
    policies = await seed_policies(session, settings.policy_rules_path)
    await seed_claims(session, settings.data_dir, policies)
    await seed_users(session, settings)
    return await count_rows(session)


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    engine = create_db_engine(settings.database_url)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            counts = await run_seed(session, settings)
            await session.commit()
    finally:
        await engine.dispose()

    width = max(len(name) for name in counts)
    lines = ["Seed complete. Rows per table:"]
    lines += [f"  {name:<{width}}  {count}" for name, count in counts.items()]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
