"""Query rewriting: follow-ups -> standalone questions, and optional multi-query."""

import logging
from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from app.rag.config import RagSettings
from app.rag.schemas import ChatTurn

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = (
    "You rewrite the user's latest message into ONE standalone question about an insurance "
    "policy, using the conversation only to resolve references (pronouns, 'and for X?', "
    "'what about it?'). Keep policy names, amounts, periods and medical terms exactly. "
    "Do not answer the question. Output only the rewritten question."
)

MULTI_QUERY_SYSTEM = (
    "Write {n} alternative phrasings of the user's question for searching insurance policy "
    "wording. Use different vocabulary (synonyms, formal policy terms) but keep the meaning "
    "and every number or amount. Output one phrasing per line, no numbering, no extra text."
)


def _text(message) -> str:  # type: ignore[no-untyped-def]
    content = message.content
    if isinstance(content, list):  # some providers return content blocks
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content).strip()


class QueryRewriter:
    def __init__(self, chat_model_factory: Callable[[], object], settings: RagSettings):
        self.chat_model_factory = chat_model_factory
        self.settings = settings

    async def rewrite(self, query: str, history: list[ChatTurn] | None) -> str:
        """Standalone version of `query`. With no history nothing is rewritten and no LLM call
        is made (saves a round trip on first questions)."""
        if not history:
            return query
        recent = history[-self.settings.max_history_turns :]
        transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)
        messages = [
            SystemMessage(REWRITE_SYSTEM),
            HumanMessage(f"Conversation:\n{transcript}\n\nLatest question: {query}"),
        ]
        try:
            reply = await self.chat_model_factory().ainvoke(messages)  # type: ignore[attr-defined]
            rewritten = _text(reply).strip('"').strip()
        except Exception:  # noqa: BLE001 - a failed rewrite must not fail the request
            logger.warning("rewrite_failed", exc_info=True)
            return query
        return rewritten or query

    async def alternatives(self, query: str) -> list[str]:
        """Extra phrasings for multi-query retrieval (empty list on any failure)."""
        n = self.settings.multi_query_count
        messages = [SystemMessage(MULTI_QUERY_SYSTEM.format(n=n)), HumanMessage(query)]
        try:
            reply = await self.chat_model_factory().ainvoke(messages)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            logger.warning("multi_query_failed", exc_info=True)
            return []
        lines = [ln.strip(' -*\t"') for ln in _text(reply).splitlines()]
        seen = {query.lower()}
        variants: list[str] = []
        for line in lines:
            if line and line.lower() not in seen:
                seen.add(line.lower())
                variants.append(line)
        return variants[:n]
