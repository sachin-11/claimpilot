"""Bearer-token configuration shared by the MCP servers (who may connect) and the client (what it
sends). Env vars, per server (`policy_docs` -> POLICY_DOCS):

  MCP_<SERVER>_TOKENS="coverage-agent=tokA,risk-agent=tokB"   identity=token pairs
  MCP_<SERVER>_TOKEN=tok                                       one token, identity "<server>-client"
"""

import os
from collections.abc import Mapping


def _env_prefix(server: str) -> str:
    return f"MCP_{server.upper()}"


def identities(server: str, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """caller identity -> token, for one server."""
    env = os.environ if env is None else env
    result: dict[str, str] = {}
    single = (env.get(f"{_env_prefix(server)}_TOKEN") or "").strip()
    if single:
        result[f"{server}-client"] = single
    for pair in (env.get(f"{_env_prefix(server)}_TOKENS") or "").split(","):
        name, sep, token = pair.partition("=")
        if sep and name.strip() and token.strip():
            result[name.strip()] = token.strip()
    return result


MIN_TOKEN_LENGTH = 16


def weak_identities(server: str, env: Mapping[str, str] | None = None) -> list[str]:
    """Callers whose token is still the `.env.example` placeholder or too short to be a secret."""
    return [
        name
        for name, token in identities(server, env).items()
        if "replace-me" in token.lower() or len(token) < MIN_TOKEN_LENGTH
    ]


def token_map(server: str, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """token -> caller identity (what a server checks incoming requests against)."""
    return {token: name for name, token in identities(server, env).items()}


def token_for(server: str, caller: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """The token a client sends: that caller's own, or the server's only/first one."""
    available = identities(server, env)
    if caller is not None:
        if caller not in available:
            raise KeyError(f"no token configured for caller {caller!r} on MCP server {server!r}")
        return available[caller]
    if not available:
        raise KeyError(
            f"no token configured for MCP server {server!r} ({_env_prefix(server)}_TOKEN)"
        )
    return next(iter(available.values()))
