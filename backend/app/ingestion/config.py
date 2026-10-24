from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import get_settings


def _default_policies_dir() -> Path:
    return get_settings().data_dir / "policies"


def _default_debug_dir() -> Path:
    return get_settings().data_dir / "debug"


class IngestionSettings(BaseSettings):
    """Ingestion knobs. Everything can be overridden from .env (CHUNK_MAX_TOKENS=600, ...),
    so Module 5 can tune chunking without code changes."""

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    policies_dir: Path = Field(default_factory=_default_policies_dir)
    debug_dir: Path = Field(default_factory=_default_debug_dir)

    # Chunking (tokens counted with tiktoken cl100k_base)
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 120
    chunk_min_tokens: int = 80

    # Regexes (case-insensitive) matched against the TOP-LEVEL section title; matching sections
    # are not ingested (e.g. ["get in touch with us"] for contact / ombudsman address lists).
    exclude_section_patterns: list[str] = Field(default_factory=list)

    # Parsing: a page with fewer extractable characters than this is treated as scanned.
    min_page_chars: int = 30

    # Embedding
    embed_batch_size: int = 64
    embed_max_batch_tokens: int = 200_000  # OpenAI allows 300k per request; stay well below
    embed_max_retries: int = 5
    embed_retry_min_seconds: float = 1.0
    embed_retry_max_seconds: float = 30.0
    # USD per 1M tokens, only used for the cost estimate in the logs (text-embedding-3-small).
    embed_cost_per_million_tokens: float = 0.02


@lru_cache
def get_ingestion_settings() -> IngestionSettings:
    return IngestionSettings()
