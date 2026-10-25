"""MCP tools per agent (least privilege). An agent is handed ONLY the tools listed here, loaded
with its own identity (token), so the MCP audit log shows which agent acted."""

import json
import logging
import uuid
from typing import Any, Protocol

import httpx
from langchain_core.tools import BaseTool

from app.agents.events import emit

logger = logging.getLogger(__name__)

AGENT_TOOLS: dict[str, dict[str, list[str]]] = {
    "intake": {
        "claims_db": [
            "get_claim",
            "get_claim_extraction",
            "get_customer_history",
            "update_claim_status",
        ]
    },
    "coverage": {"policy_docs": ["search_policy", "get_policy_rules"]},
    "risk": {
        "claims_db": [
            "get_claim",
            "get_claim_extraction",
            "get_customer_history",
            "list_claims",
        ]
    },
    "decision": {"claims_db": ["save_decision_draft", "update_claim_status"]},
    # Used by the supervisor's error path only: mark the claim failed.
    "supervisor": {"claims_db": ["update_claim_status"]},
    "finalize": {"notify": ["send_status_update"]},
}


# MCP identity (token name) each agent connects as; default "<agent>-agent".
AGENT_IDENTITY: dict[str, str] = {"finalize": "notify-agent"}


class ToolFailure(RuntimeError):
    """A tool answered with an error (claim not found, transition refused, ...). Not transient."""


class ToolProvider(Protocol):
    async def for_agent(self, agent: str) -> list[BaseTool]: ...


class McpToolProvider:
    """Loads each agent's tools from the MCP servers (cached per agent)."""

    def __init__(self) -> None:
        self._cache: dict[str, list[BaseTool]] = {}

    async def for_agent(self, agent: str) -> list[BaseTool]:
        if agent not in self._cache:
            from app.mcp_client.client import get_mcp_tools

            self._cache[agent] = await get_mcp_tools(
                AGENT_TOOLS[agent], caller=AGENT_IDENTITY.get(agent, f"{agent}-agent")
            )
        return self._cache[agent]


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: network trouble and provider 5xx / rate limits, never a refused request."""
    if isinstance(exc, ToolFailure):
        return False
    if isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "McpError",
        "ExceptionGroup",
        "CircuitOpenError",
    }


def parse_tool_message(message: Any) -> Any:
    """The structured result of a tool call (artifact first, else the JSON text blocks)."""
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, dict) and artifact.get("structured_content") is not None:
        data = artifact["structured_content"]
        return data["result"] if isinstance(data, dict) and set(data) == {"result"} else data
    content = message.content
    if isinstance(content, str):
        return json.loads(content)
    items = [
        json.loads(block["text"])
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return items[0] if len(items) == 1 else items


def summarize_args(args: dict[str, Any], limit: int = 80) -> dict[str, Any]:
    """Short argument summary for events (no long free text)."""
    return {
        k: (v if not isinstance(v, str) or len(v) <= limit else v[:limit] + "...")
        for k, v in args.items()
        if k != "idempotency_key"
    }


class ToolBox:
    """The tools one agent may use, with event emission and error mapping."""

    def __init__(self, agent: str, tools: list[BaseTool]) -> None:
        self.agent = agent
        self.tools = {t.name: t for t in tools}

    @property
    def names(self) -> set[str]:
        return set(self.tools)

    def langchain_tools(self) -> list[BaseTool]:
        return list(self.tools.values())

    async def call(self, name: str, args: dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise PermissionError(f"the {self.agent} agent has no tool {name!r}")
        emit(
            {"type": "tool_called", "agent": self.agent, "tool": name, "args": summarize_args(args)}
        )
        message = await tool.ainvoke(
            {"type": "tool_call", "name": name, "args": args, "id": uuid.uuid4().hex}
        )
        if getattr(message, "status", None) == "error":
            content = message.content
            text = (
                content
                if isinstance(content, str)
                else " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            )
            logger.warning("tool_error", extra={"agent": self.agent, "tool": name})
            raise ToolFailure(text.removeprefix(f"Error executing tool {name}: ")[:300])
        return parse_tool_message(message)


async def toolbox_for(provider: ToolProvider, agent: str) -> ToolBox:
    return ToolBox(agent, await provider.for_agent(agent))
