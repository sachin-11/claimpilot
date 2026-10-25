"""Tool errors. A ToolError message is shown to the calling LLM, so it says what was wrong and
what to do instead; it never contains a stack trace or internal detail."""

from mcp.server.fastmcp.exceptions import ToolError

__all__ = ["ToolError", "claim_not_found", "policy_not_found"]


def claim_not_found(claim_number: str) -> ToolError:
    return ToolError(
        f"Claim {claim_number!r} not found. Use list_claims to see valid claim numbers."
    )


def policy_not_found(policy_code: str) -> ToolError:
    return ToolError(
        f"Policy {policy_code!r} not found. Use list_policies to see valid policy codes."
    )
