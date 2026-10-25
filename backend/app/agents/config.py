from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Agent workflow knobs (Module 8). All overridable from .env."""

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Whole-graph step limit (a LangGraph "super-step" is one round of node executions).
    agent_recursion_limit: int = 40
    # Tool-call budgets of the agent subgraphs (each call is a model step + a tool step).
    coverage_max_tool_calls: int = 8
    risk_max_tool_calls: int = 4
    # Risk score (0-100) at or above which the decision becomes flag_for_review.
    risk_review_threshold: int = 50
    # Admission this many days after the policy start is flagged.
    early_admission_days: int = 60
    # More than this many OTHER claims on the policy within the window is "frequent".
    frequent_claims_count: int = 2
    frequent_claims_window_days: int = 365
    # The supervisor LLM is skipped when only one step is allowed.
    supervisor_use_llm: bool = True
    # Retries (LangGraph RetryPolicy) for transient LLM / MCP errors before the error node.
    node_retry_attempts: int = 3
    node_retry_initial_interval: float = 1.0
    # Where AsyncPostgresSaver connects; empty = DATABASE_URL without the asyncpg driver suffix.
    checkpoint_database_url: str = ""
    # Notification channel used when a run is finalised.
    notify_channel: str = "email"


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()
