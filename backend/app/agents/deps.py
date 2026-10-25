"""Everything the nodes need, injectable (tests pass fakes)."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.config import AgentSettings, get_agent_settings
from app.agents.tools import ToolProvider


class IntakeRunner(Protocol):
    async def process(self, claim_id: UUID) -> Any: ...


@dataclass
class AgentDeps:
    tools: ToolProvider
    chat_model: Callable[[], Any]  # get_chat_model(): OpenAI primary, Groq fallback
    session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]]
    data_dir: Path
    intake: IntakeRunner | None = None
    settings: AgentSettings = field(default_factory=get_agent_settings)
