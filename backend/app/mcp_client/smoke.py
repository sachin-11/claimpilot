"""`python -m app.mcp_client.smoke`: list every tool on the three HTTP MCP servers, then call
search_policy, get_claim and get_claim_extraction for CLM-001."""

import asyncio
import logging
import sys

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.mcp_client.client import SERVERS, build_client, get_mcp_tools

logger = logging.getLogger("mcp.smoke")


def _show(title: str, value: object, limit: int = 1200) -> None:
    text = str(value)
    logger.info("%s: %s%s", title, text[:limit], " ..." if len(text) > limit else "")


async def main() -> int:
    setup_logging(get_settings().log_level)
    failures = 0

    client = build_client()
    for server in SERVERS:
        try:
            tools = await client.get_tools(server_name=server)
        except Exception as exc:
            failures += 1
            logger.error("%s: cannot list tools (%s: %s)", server, type(exc).__name__, exc)
            continue
        for t in tools:
            logger.info("%s.%s - %s", server, t.name, (t.description or "").split("\n")[0][:90])

    tools = {
        t.name: t
        for t in await get_mcp_tools(
            {
                "policy_docs": ["search_policy"],
                "claims_db": ["get_claim", "get_claim_extraction"],
            }
        )
    }
    calls = [
        ("search_policy", {"query": "cataract waiting period", "top_n": 2}),
        ("get_claim", {"claim_number": "CLM-001"}),
        ("get_claim_extraction", {"claim_number": "CLM-001"}),
    ]
    for name, args in calls:
        try:
            _show(f"{name}({args})", await tools[name].ainvoke(args))
        except Exception as exc:
            failures += 1
            logger.error("%s failed (%s: %s)", name, type(exc).__name__, exc)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
