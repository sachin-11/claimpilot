from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def has_real_key(key: SecretStr | None) -> bool:
    """True if a key is set and is not the empty/placeholder value from .env.example."""
    if key is None:
        return False
    value = key.get_secret_value().strip()
    return bool(value) and not value.lower().endswith("replace-me")


# backend/app/core/config.py -> repo root on the host, "/" in the container (code lives in /app).
_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Later files win: backend/.env overrides the repo-root .env (docker uses real env vars).
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://claimpilot:claimpilot@localhost:5432/claimpilot"
    # Optional: separate database for pytest (defaults to "<database name>_test").
    test_database_url: str | None = None
    redis_url: str = "redis://localhost:6379/0"

    # Seed data (Module 2). Stored paths in the DB are relative to data_dir.
    data_dir: Path = _ROOT / "data"
    policy_rules_path: Path = _ROOT / "data_gen" / "policy_rules.yaml"
    seed_admin_email: str = "admin@example.com"
    seed_admin_password: SecretStr | None = None
    seed_officer_email: str = "officer@example.com"
    seed_officer_password: SecretStr | None = None

    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Observability (Module 10). Tracing is optional: without both keys the app runs untraced.
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = ""  # http://langfuse-web:3000 (self-hosted) or a Langfuse Cloud URL
    langfuse_host: str = ""  # older name of the same setting
    usd_inr_rate: float = 88.0  # for llm_cost_inr_total and the daily budget; set it yourself
    primary_model: str = "gpt-4o-mini"
    fallback_model: str = "llama-3.3-70b-versatile"
    # Used by the evaluation runs (eval/e2e, eval/rag), see `build_anthropic_chat_model`.
    anthropic_model: str = "claude-haiku-5-5"
    embedding_model: str = "text-embedding-3-small"

    # Comma-separated in the env var; use `cors_origin_list` in code.
    cors_origins: str = "http://localhost:5173"

    # Security (Module 10). The default secret works for development only: `validate_for_production`
    # refuses to start with it when APP_ENV=production.
    jwt_secret: SecretStr = SecretStr("dev-insecure-jwt-secret-change-me")
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 7
    login_max_failures: int = 5  # per email + IP ...
    login_window_minutes: int = 15  # ... in this window
    metrics_token: SecretStr | None = None  # bearer token for Prometheus; admins can scrape as well

    # Rate limits (requests per minute). LLM-heavy endpoints get the stricter pair.
    rate_limit_enabled: bool = True
    rate_limit_in_tests: bool = False  # off when APP_ENV=test unless a test switches it on
    rate_limit_user_per_minute: int = 120
    rate_limit_ip_per_minute: int = 300
    rate_limit_llm_user_per_minute: int = 10
    rate_limit_llm_ip_per_minute: int = 30
    # Above this estimated spend (INR) per day new LLM work is refused. 0 switches the guard off.
    daily_llm_budget_inr: float = 500.0

    # Agent runs, extraction and ingestion run in the arq worker (Redis queue) when true; in the
    # API process otherwise (tests, the evaluation harness and a single-process dev setup).
    use_worker: bool = False
    # /health/ready also pings the MCP servers and the worker heartbeat (off in tests, which
    # have neither; the checks themselves are tested one by one).
    readiness_extra_checks: bool = True
    worker_max_jobs: int = 4
    worker_job_timeout_seconds: int = 1200  # one agent run, extraction or ingestion
    worker_shutdown_wait_seconds: int = 120  # on SIGTERM: let running jobs finish this long
    worker_max_tries: int = 4  # per job; retries back off 5 s, 10 s, 20 s
    worker_metrics_port: int = 9102

    llm_max_retries: int = 2
    llm_timeout_seconds: float = 30.0
    breaker_failure_threshold: int = 3
    breaker_window_seconds: float = 60.0
    breaker_cooldown_seconds: float = 30.0

    outbound_timeout_seconds: float = 30.0  # embeddings, MCP calls, OCR provider
    reranker_timeout_seconds: float = 15.0
    cors_allow_credentials: bool = True

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class InsecureConfigurationError(RuntimeError):
    """APP_ENV=production with missing secrets or development defaults."""


def production_problems(settings: Settings) -> list[str]:
    """Everything that must not reach production (empty list = fine)."""
    problems: list[str] = []
    secret = settings.jwt_secret.get_secret_value()
    if secret == "dev-insecure-jwt-secret-change-me" or len(secret) < 32:
        problems.append("JWT_SECRET is the development default or shorter than 32 characters")
    if not has_real_key(settings.openai_api_key):
        problems.append("OPENAI_API_KEY is not set")
    if "claimpilot:claimpilot@" in settings.database_url:
        problems.append("DATABASE_URL still uses the default claimpilot:claimpilot credentials")
    origins = settings.cors_origin_list
    if not origins or "*" in origins or any("localhost" in o or "127.0.0.1" in o for o in origins):
        problems.append("CORS_ORIGINS is empty, '*' or points at localhost")
    for name in ("seed_admin_password", "seed_officer_password"):
        value = getattr(settings, name)
        if value is not None and value.get_secret_value() in {
            "admin",
            "officer",
            "password",
            "changeme",
        }:
            problems.append(f"{name.upper()} is a well-known default")
    return problems


def validate_for_production(settings: Settings) -> None:
    """Fail fast at startup in production: a half-configured service must not come up."""
    if not settings.is_production:
        return
    problems = production_problems(settings)
    if problems:
        raise InsecureConfigurationError(
            "refusing to start with APP_ENV=production: " + "; ".join(problems)
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
