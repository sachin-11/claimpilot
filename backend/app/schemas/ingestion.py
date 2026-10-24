from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import IngestionStatus
from app.schemas.base import ORMModel


class IngestRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    policy_code: str | None = None
    all_policies: bool = Field(default=False, alias="all")
    force: bool = False

    @model_validator(mode="after")
    def exactly_one_target(self) -> "IngestRequest":
        if bool(self.policy_code) == self.all_policies:
            raise ValueError("send either policy_code or all=true")
        return self


class IngestRunRead(ORMModel):
    id: UUID
    policy_id: UUID
    file_sha256: str | None
    status: IngestionStatus
    chunks_total: int
    chunks_new: int
    chunks_deleted: int
    tokens_used: int
    error: str | None
    started_at: datetime
    finished_at: datetime | None


class IngestRunRef(BaseModel):
    run_id: UUID
    policy_code: str


class IngestStarted(BaseModel):
    run_id: UUID  # the first run; for all=true see `runs`
    runs: list[IngestRunRef]


class PolicySummary(BaseModel):
    policy_code: str
    name: str
    source_pdf: str | None
    version: int
    is_active: bool
    chunk_count: int
    last_run: IngestRunRead | None
