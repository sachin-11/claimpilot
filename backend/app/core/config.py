from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def has_real_key(key: SecretStr | None) -> bool:
    """True if a key is set and is not the empty/placeholder value from .env.example."""
    if key is None:
        return False
    value = key.get_secret_value().strip()
    return bool(value) and not value.lower().endswith("replace-me")


class Settings(BaseSettings):
    # Later files win: backend/.env overrides the repo-root .env (docker uses real env vars).
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://claimpilot:claimpilot@localhost:5432/claimpilot"
    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "text-embedding-3-small"

    # Comma-separated in the env var; use `cors_origin_list` in code.
    cors_origins: str = "http://localhost:5173"

    llm_max_retries: int = 2
    llm_timeout_seconds: float = 30.0
    breaker_failure_threshold: int = 3
    breaker_window_seconds: float = 60.0
    breaker_cooldown_seconds: float = 30.0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
