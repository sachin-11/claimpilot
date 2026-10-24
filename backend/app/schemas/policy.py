from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.base import ORMModel, ReadModel


class PolicyCreate(BaseModel):
    policy_code: str
    name: str
    insurer: str | None = None
    source_pdf: str | None = None
    rules: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    is_active: bool = True


class PolicyRead(PolicyCreate, ReadModel):
    pass


class PolicyChunkCreate(BaseModel):
    chunk_index: int
    content: str
    content_hash: str
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    token_count: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class PolicyChunkRead(ReadModel):
    """No embedding on purpose: 1536 floats per chunk are never useful in API output."""

    policy_id: UUID
    chunk_index: int
    content: str
    content_hash: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    token_count: int | None
    meta: dict[str, Any]


class BulkUpsertResult(ORMModel):
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
