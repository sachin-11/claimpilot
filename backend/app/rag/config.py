from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    """Retrieval / answering knobs. Every field can be overridden from .env (VECTOR_TOP_K=30)."""

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Retrieval
    vector_top_k: int = 20
    keyword_top_k: int = 20
    fused_top_k: int = 20
    rrf_k: int = 60
    hnsw_ef_search: int = 100  # candidates the HNSW index looks at before filters apply
    max_keyword_terms: int = 12

    # Reranking
    reranker: Literal["none", "local", "cohere"] = "none"
    rerank_top_n: int = 5
    cohere_api_key: SecretStr | None = None
    cohere_rerank_model: str = "rerank-v3.5"
    local_rerank_model: str = "BAAI/bge-reranker-base"
    rerank_max_chars: int = 2000  # clause text sent to the reranker per candidate

    # Query rewriting
    multi_query: bool = False
    multi_query_count: int = 2
    max_history_turns: int = 6  # last N chat messages used to rewrite a follow-up

    # Answering
    max_citation_text_chars: int = 200

    # Semantic cache
    cache_enabled: bool = True
    cache_sim_threshold: float = 0.95
    cache_ttl_seconds: int = 86_400
    cache_max_entries_per_filter: int = 200
    cache_prefix: str = "rag:cache"


@lru_cache
def get_rag_settings() -> RagSettings:
    return RagSettings()
