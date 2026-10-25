"""Facts that only the structured claim form carries (not part of the Module 6 extraction):
declared pre-existing conditions and whether the admission was an accident."""

import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ClaimDocument, DocType

logger = logging.getLogger(__name__)


def facts_from_form(form: dict[str, Any]) -> dict[str, Any]:
    conditions = form.get("pre_existing_conditions") or []
    return {
        "pre_existing_conditions": [str(c)[:120] for c in conditions][:10],
        "is_accident": bool(form.get("is_accident_case")),
        "known": True,
    }


async def load_claim_form_facts(
    session: AsyncSession, data_dir: Path, claim_id: UUID
) -> dict[str, Any]:
    """From the claim's JSON claim form. {"known": False} when there is none (e.g. a PDF form)."""
    rows = await session.execute(
        select(ClaimDocument).where(
            ClaimDocument.claim_id == claim_id, ClaimDocument.doc_type == DocType.CLAIM_FORM
        )
    )
    for doc in rows.scalars():
        if not doc.mime_type.endswith("json"):
            continue
        try:
            form = json.loads((data_dir / doc.file_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("claim_form_unreadable", extra={"claim_id": str(claim_id)})
            continue
        return facts_from_form(form)
    return {"pre_existing_conditions": [], "is_accident": False, "known": False}
