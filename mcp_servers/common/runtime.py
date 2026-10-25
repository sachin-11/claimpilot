"""`python -m mcp_servers.<name>.server --transport stdio|http --port N`."""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from mcp.server.fastmcp import FastMCP

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.mcp_client.tokens import token_map
from mcp_servers.common.auth import BearerAuthMiddleware

logger = logging.getLogger("mcp.runtime")


def run_cli(name: str, factory: Callable[..., FastMCP], argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog=f"mcp_servers.{name}.server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="http only")
    parser.add_argument("--port", type=int, default=8000, help="http only")
    args = parser.parse_args(argv)

    setup_logging(get_settings().log_level)
    if args.transport == "stdio":
        # stdout carries the protocol, so logs must go to stderr.
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(sys.stderr)
        factory().run("stdio")
        return

    tokens = token_map(name)
    if not tokens:
        sys.exit(
            f"refusing to start: set MCP_{name.upper()}_TOKEN (or MCP_{name.upper()}_TOKENS) "
            "for http mode"
        )
    import uvicorn

    app = factory(host=args.host).streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, tokens=tokens)
    logger.info("mcp_http_start", extra={"server": name, "port": args.port, "callers": len(tokens)})
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
