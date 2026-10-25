"""Settings shared by the MCP servers. Everything comes from env vars (see mcp_client/tokens.py
for the bearer tokens)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class McpSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Write tools are not even registered when this is false (the safe default).
    write_enabled: bool = False
    # Caller identity in stdio mode, where there is no bearer token.
    mcp_caller_id: str = "stdio-user"
    notify_max_per_claim_per_hour: int = 5
    # Longest clause text returned by policy_docs tools / resources.
    max_clause_chars: int = 1500


@lru_cache
def get_mcp_settings() -> McpSettings:
    return McpSettings()
