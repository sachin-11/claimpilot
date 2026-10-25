"""policy_docs MCP server (read-only): policy wordings, clause search, grounded answers."""

import json
from typing import Annotated, Any
from uuid import UUID

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from sqlalchemy import select

from app.models import Policy, PolicyChunk
from app.rag.schemas import RetrievedChunk
from app.repositories.policy import PolicyChunkRepository, PolicyRepository
from mcp_servers.common.deps import ServerDeps, default_deps
from mcp_servers.common.errors import ToolError, policy_not_found
from mcp_servers.common.instrument import instrumented
from mcp_servers.common.runtime import run_cli

NAME = "policy_docs"

INSTRUCTIONS = (
    "Read-only access to health-insurance policy wordings. Use search_policy to find the clauses "
    "that apply to a question and quote their section_path; use answer_policy_question for a "
    "ready-made grounded answer. Never state a coverage rule that is not in a returned clause."
)

PolicyCode = Annotated[
    str | None,
    Field(
        description="Policy to search, e.g. 'POL-AS'. Omit to search all policies.",
        max_length=50,
    ),
]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _clause(chunk: RetrievedChunk, limit: int) -> dict[str, Any]:
    score = chunk.rerank_score if chunk.rerank_score is not None else chunk.fused_score
    return {
        "chunk_id": str(chunk.chunk_id),
        "policy_code": chunk.policy_code,
        "section_path": chunk.section_path,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "content": _clip(chunk.content, limit),
        "score": round(score, 4),
    }


def create_server(deps: ServerDeps | None = None, *, host: str = "127.0.0.1") -> FastMCP:
    deps = deps or default_deps()
    limit = deps.settings.max_clause_chars
    mcp = FastMCP(NAME, instructions=INSTRUCTIONS, host=host)
    tool = instrumented(NAME, deps.settings)

    async def require_policy(session: Any, code: str) -> Policy:
        policy = await PolicyRepository(session).get_by_code(code)
        if policy is None:
            raise policy_not_found(code)
        return policy

    @mcp.tool()
    @tool
    async def list_policies(ctx: Context) -> list[dict[str, Any]]:
        """List the insurance policies that have wordings available, with how many searchable
        clauses each has. Call this first to learn valid policy_code values."""
        async with deps.session() as session:
            chunks = PolicyChunkRepository(session)
            return [
                {
                    "policy_code": p.policy_code,
                    "name": p.name,
                    "insurer": p.insurer,
                    "chunk_count": await chunks.count_by_policy(p.id),
                }
                for p in await PolicyRepository(session).list_active()
            ]

    @mcp.tool()
    @tool
    async def search_policy(
        ctx: Context,
        query: Annotated[str, Field(min_length=3, max_length=500, description="What to look up")],
        policy_code: PolicyCode = None,
        top_n: Annotated[int, Field(ge=1, le=10, description="Number of clauses")] = 5,
    ) -> list[dict[str, Any]]:
        """Find the policy clauses most relevant to a question (hybrid vector + keyword search,
        reranked). Returns clauses with section_path and pages for citation, best first. Does
        not call an LLM. Use this to check waiting periods, exclusions, sub-limits and so on."""
        if policy_code is not None:
            async with deps.session() as session:
                await require_policy(session, policy_code)
        codes = [policy_code] if policy_code else None
        chunks = await deps.rag().search(query, policy_codes=codes, top_n=top_n)
        return [_clause(c, limit) for c in chunks]

    @mcp.tool()
    @tool
    async def answer_policy_question(
        ctx: Context,
        question: Annotated[str, Field(min_length=3, max_length=500)],
        policy_code: PolicyCode = None,
    ) -> dict[str, Any]:
        """Answer a coverage question in plain English, grounded in the policy text with
        citations. If the wording does not settle it, insufficient_context is true. Slower and
        costlier than search_policy; prefer search_policy when you only need the clauses."""
        if policy_code is not None:
            async with deps.session() as session:
                await require_policy(session, policy_code)
        result = await deps.rag().answer(
            question, policy_codes=[policy_code] if policy_code else None
        )
        return {
            "answer": result.answer,
            "citations": [
                {
                    "chunk_id": str(c.chunk_id),
                    "section_path": c.section_path,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "supporting_text": c.supporting_text,
                }
                for c in result.citations
            ],
            "insufficient_context": result.insufficient_context,
            "confidence": result.confidence,
        }

    @mcp.tool()
    @tool
    async def get_policy_rules(
        ctx: Context,
        policy_code: Annotated[str, Field(min_length=1, max_length=50)],
    ) -> dict[str, Any]:
        """Structured rules of a policy (waiting periods, sub-limits, co-pay, exclusions) as
        configured by the insurer team. May differ from the PDF wording: when the two disagree,
        the clause text from search_policy is what to cite."""
        async with deps.session() as session:
            policy = await require_policy(session, policy_code)
            return {
                "policy_code": policy.policy_code,
                "name": policy.name,
                "version": policy.version,
                "rules": policy.rules,
            }

    @mcp.resource("policy://{policy_code}/sections", mime_type="application/json")
    async def policy_sections(policy_code: str) -> str:
        """Section paths of a policy wording, in document order."""
        async with deps.session() as session:
            policy = await require_policy(session, policy_code)
            rows = await session.execute(
                select(PolicyChunk.section_path)
                .where(PolicyChunk.policy_id == policy.id)
                .order_by(PolicyChunk.chunk_index)
            )
            seen: dict[str, None] = {}
            for (path,) in rows:
                if path:
                    seen.setdefault(path)
            return json.dumps(list(seen))

    @mcp.resource("policy://{policy_code}/chunk/{chunk_id}", mime_type="application/json")
    async def policy_chunk(policy_code: str, chunk_id: str) -> str:
        """One policy clause with its metadata (section, pages, clause numbers)."""
        try:
            chunk_uuid = UUID(chunk_id)
        except ValueError as exc:
            raise ToolError(f"chunk_id {chunk_id!r} is not a valid id.") from exc
        async with deps.session() as session:
            policy = await require_policy(session, policy_code)
            chunk = (
                await session.execute(
                    select(PolicyChunk).where(
                        PolicyChunk.id == chunk_uuid, PolicyChunk.policy_id == policy.id
                    )
                )
            ).scalar_one_or_none()
            if chunk is None:
                raise ToolError(f"Clause {chunk_id} not found in policy {policy_code}.")
            meta = {
                k: v
                for k, v in chunk.meta.items()
                if k in ("clauses", "headings", "has_table", "block_types", "insurer")
            }
            return json.dumps(
                {
                    "chunk_id": str(chunk.id),
                    "policy_code": policy.policy_code,
                    "section_path": chunk.section_path,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "content": _clip(chunk.content, limit * 2),
                    "metadata": meta,
                }
            )

    return mcp


def main() -> None:
    run_cli(NAME, create_server)


if __name__ == "__main__":
    main()
