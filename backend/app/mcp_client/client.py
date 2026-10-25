"""LangChain tools from the three ClaimPilot MCP servers (HTTP), for the Module 8 agents.

Least privilege: an agent asks only for the tools it needs, e.g.

    tools = await get_mcp_tools({"policy_docs": ["search_policy"]}, caller="coverage-agent")

Server URLs: MCP_<SERVER>_URL (default: the docker-compose service). Tokens: see tokens.py.
"""

import os
from collections.abc import Iterable, Mapping

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.mcp_client.tokens import token_for

SERVERS: dict[str, str] = {
    "policy_docs": "http://mcp-policy-docs:8101/mcp",
    "claims_db": "http://mcp-claims-db:8102/mcp",
    "notify": "http://mcp-notify:8103/mcp",
}


def server_url(server: str) -> str:
    return os.environ.get(f"MCP_{server.upper()}_URL", SERVERS[server])


def build_client(
    servers: Iterable[str] | None = None, caller: str | None = None
) -> MultiServerMCPClient:
    names = list(servers) if servers is not None else list(SERVERS)
    unknown = [n for n in names if n not in SERVERS]
    if unknown:
        raise ValueError(f"unknown MCP server(s) {unknown}; choose from {sorted(SERVERS)}")
    return MultiServerMCPClient(
        {
            name: {
                "transport": "streamable_http",
                "url": server_url(name),
                "headers": {"Authorization": f"Bearer {token_for(name, caller)}"},
            }
            for name in names
        }
    )


async def get_mcp_tools(
    allow: Mapping[str, Iterable[str] | None] | None = None, caller: str | None = None
) -> list[BaseTool]:
    """Tools from the servers named in `allow`, each limited to the listed tool names
    (None = every tool of that server). `allow=None` loads everything from all three servers.

    `caller` picks which configured identity's token is sent (it shows up in audit logs)."""
    allow = allow if allow is not None else dict.fromkeys(SERVERS)
    client = build_client(allow.keys(), caller)
    tools: list[BaseTool] = []
    for server, names in allow.items():
        server_tools = await client.get_tools(server_name=server)
        if names is not None:
            wanted = set(names)
            missing = wanted - {t.name for t in server_tools}
            if missing:
                raise ValueError(f"MCP server {server!r} has no tool(s) {sorted(missing)}")
            server_tools = [t for t in server_tools if t.name in wanted]
        tools.extend(server_tools)
    return tools
