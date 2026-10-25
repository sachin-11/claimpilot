"""notify MCP server: customer status messages. Delivery is mocked: a message is rendered and
stored in `notifications` (status sent_mock), nothing leaves the system."""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from jinja2 import FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from sqlalchemy import func, select, text

from app.models import (
    Claim,
    ClaimStatus,
    Customer,
    CustomerPolicy,
    Notification,
    NotificationChannel,
    NotificationStatus,
)
from app.repositories.claim import ClaimRepository
from mcp_servers.common.audit import audit_agent_action
from mcp_servers.common.deps import ServerDeps, default_deps
from mcp_servers.common.errors import ToolError, claim_not_found
from mcp_servers.common.instrument import caller_id, instrumented
from mcp_servers.common.runtime import run_cli

NAME = "notify"
logger = logging.getLogger("mcp.notify")

TEMPLATE_DIR = Path(__file__).parent / "templates"
Template = Literal["received", "documents_requested", "approved", "partially_approved", "rejected"]

# A final-outcome message is only sent when the claim really is in that state, so an agent
# cannot tell a customer "approved" before a human has approved.
REQUIRED_STATUS: dict[str, ClaimStatus] = {
    "documents_requested": ClaimStatus.DOCUMENTS_REQUESTED,
    "approved": ClaimStatus.APPROVED,
    "partially_approved": ClaimStatus.PARTIALLY_APPROVED,
    "rejected": ClaimStatus.REJECTED,
}

# `extra` keys a template may use. Anything else (e.g. "to", "email", "phone") is ignored.
EXTRA_KEYS = {"missing_documents", "reason"}

INSTRUCTIONS = (
    "Sends status updates to the customer of a claim. The recipient is always the claim's "
    "customer on record; it cannot be chosen. Final-outcome templates (approved, "
    "partially_approved, rejected) work only after a human has set that status."
)

_env = SandboxedEnvironment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def mask_recipient(channel: NotificationChannel, value: str) -> str:
    if channel == NotificationChannel.EMAIL:
        local, _, domain = value.partition("@")
        return f"{local[:1]}***@{domain}"
    return "*" * max(len(value) - 4, 0) + value[-4:]


def sanitize_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only the allowed keys with simple, bounded values."""
    clean: dict[str, Any] = {}
    for key, value in (extra or {}).items():
        if key not in EXTRA_KEYS:
            continue
        if key == "missing_documents":
            if not isinstance(value, list) or len(value) > 10:
                raise ToolError("extra.missing_documents must be a list of at most 10 names.")
            clean[key] = [str(v).strip()[:100] for v in value if str(v).strip()]
        else:
            clean[key] = str(value).strip()[:300]
    return clean


def render_message(template: str, channel: NotificationChannel, context: dict[str, Any]) -> str:
    return _env.get_template(f"{template}.j2").render(channel=channel.value, **context).strip()


def create_server(deps: ServerDeps | None = None, *, host: str = "127.0.0.1") -> FastMCP:
    deps = deps or default_deps()
    mcp = FastMCP(NAME, instructions=INSTRUCTIONS, host=host)
    if not deps.settings.write_enabled:
        return mcp  # notify only writes: nothing to register
    tool = instrumented(NAME, deps.settings)
    max_per_hour = deps.settings.notify_max_per_claim_per_hour

    @mcp.tool()
    @tool
    async def send_status_update(
        ctx: Context,
        claim_number: Annotated[
            str, Field(min_length=3, max_length=50, pattern=r"^[A-Za-z0-9._-]+$")
        ],
        channel: Literal["email", "sms"],
        template: Template,
        idempotency_key: Annotated[str, Field(min_length=8, max_length=128)],
        extra: Annotated[
            dict[str, Any] | None,
            Field(
                description="Optional template values: missing_documents (list of names) and "
                "reason (short text). Other keys are ignored; the recipient cannot be set."
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Send the customer of a claim a status message by email or SMS (delivery is mocked).
        Templates: received, documents_requested, approved, partially_approved, rejected. The
        recipient is the claim's customer on record. approved / partially_approved / rejected
        are sent only if the claim already has that status (set by a human reviewer).
        Idempotent per idempotency_key; limited to a few messages per claim per hour."""
        caller = caller_id(ctx, deps.settings)
        ignored = sorted(set(extra or {}) - EXTRA_KEYS)
        if ignored:
            logger.warning("notify_extra_ignored", extra={"keys": ignored, "caller": caller})
        channel_enum = NotificationChannel(channel)

        async with deps.session() as session:
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:k, 0))"),
                {"k": f"notify:{idempotency_key}"},
            )
            claim = await ClaimRepository(session).get_by_number(claim_number)
            if claim is None:
                raise claim_not_found(claim_number)

            existing = (
                await session.execute(
                    select(Notification).where(Notification.idempotency_key == idempotency_key)
                )
            ).scalar_one_or_none()
            if existing is not None:
                if (existing.claim_id, existing.channel.value, existing.template) != (
                    claim.id,
                    channel,
                    template,
                ):
                    raise ToolError(
                        f"idempotency_key {idempotency_key!r} was already used for a different "
                        "message. Use a new key."
                    )
                return _result(existing, claim, replay=True)

            required = REQUIRED_STATUS.get(template)
            if required is not None and claim.status != required:
                raise ToolError(
                    f"Cannot send '{template}': claim {claim_number} is '{claim.status.value}', "
                    f"not '{required.value}'."
                )

            customer = await _customer_of(session, claim)
            recipient = (
                customer.email if channel_enum == NotificationChannel.EMAIL else customer.phone
            )
            if not recipient:
                raise ToolError(
                    f"The customer of claim {claim_number} has no {channel} contact on record."
                )

            since = datetime.now(UTC) - timedelta(hours=1)
            recent = (
                await session.execute(
                    select(func.count())
                    .select_from(Notification)
                    .where(Notification.claim_id == claim.id, Notification.created_at >= since)
                )
            ).scalar_one()
            if recent >= max_per_hour:
                raise ToolError(
                    f"Rate limit: claim {claim_number} already had {recent} notifications in the "
                    f"last hour (max {max_per_hour}). Try again later."
                )

            safe_extra = sanitize_extra(extra)
            message = render_message(
                template,
                channel_enum,
                {
                    "name": customer.full_name.split()[0] if customer.full_name else "Customer",
                    "claim_number": claim.claim_number,
                    "claimed_amount": _rupees(claim.claimed_amount),
                    "approved_amount": _rupees(claim.approved_amount),
                    "missing_documents": safe_extra.get("missing_documents"),
                    "reason": safe_extra.get("reason"),
                },
            )
            # Mock delivery: the stored row is the "sent" message.
            row = Notification(
                claim_id=claim.id,
                channel=channel_enum,
                template=template,
                rendered_message=message,
                status=NotificationStatus.SENT_MOCK,
                idempotency_key=idempotency_key,
            )
            session.add(row)
            await session.flush()
            await audit_agent_action(
                session,
                caller=caller,
                action="notification.sent_mock",
                entity_type="notification",
                entity_id=row.id,
                details={
                    "claim_number": claim.claim_number,
                    "channel": channel,
                    "template": template,
                },
            )
            result = _result(
                row, claim, replay=False, recipient=mask_recipient(channel_enum, recipient)
            )
            await session.commit()
            return result

    return mcp


def _rupees(value: Any) -> str | None:
    return None if value is None else f"{value:,.2f}"


async def _customer_of(session: Any, claim: Claim) -> Customer:
    if claim.customer_policy_id is None:
        raise ToolError(f"Claim {claim.claim_number} is not linked to a customer.")
    customer = (
        await session.execute(
            select(Customer)
            .join(CustomerPolicy, CustomerPolicy.customer_id == Customer.id)
            .where(CustomerPolicy.id == claim.customer_policy_id)
        )
    ).scalar_one_or_none()
    if customer is None:
        raise ToolError(f"Claim {claim.claim_number} is not linked to a customer.")
    return customer


def _result(
    row: Notification, claim: Claim, *, replay: bool, recipient: str | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "notification_id": str(row.id),
        "claim_number": claim.claim_number,
        "channel": row.channel.value,
        "template": row.template,
        "status": row.status.value,
        "idempotent_replay": replay,
    }
    if recipient:
        out["recipient"] = recipient
    return out


def main() -> None:
    run_cli(NAME, create_server)


if __name__ == "__main__":
    main()
