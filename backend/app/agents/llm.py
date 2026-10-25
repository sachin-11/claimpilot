"""Structured LLM calls for the agent nodes. Models come from `get_chat_model()` (OpenAI primary,
Groq fallback); the model that actually answered is reported so it can go to `models_used`."""

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """The model answered but not with the requested structure."""


def model_name_of(message: Any) -> str | None:
    meta = getattr(message, "response_metadata", None) or {}
    return meta.get("model_name") or meta.get("model")


async def ask_structured(
    chat_model: Any, schema: type[T], messages: Sequence[BaseMessage], *, retries: int = 1
) -> tuple[T, str | None]:
    """One structured answer. An invalid answer is retried `retries` times with the validation
    error shown to the model; transport errors are NOT caught here (the node's RetryPolicy
    handles them)."""
    runnable = chat_model.with_structured_output(
        schema, method="function_calling", include_raw=True
    )
    convo = list(messages)
    for attempt in range(retries + 1):
        out = await runnable.ainvoke(convo)
        parsed = out.get("parsed")
        if parsed is not None:
            return parsed, model_name_of(out.get("raw"))
        error = str(out.get("parsing_error"))[:600]
        logger.warning(
            "llm_invalid_structure", extra={"schema": schema.__name__, "attempt": attempt}
        )
        convo = [
            *messages,
            HumanMessage(
                content=f"Your previous answer was not valid ({error}). "
                f"Answer again, following the schema exactly."
            ),
        ]
    raise LLMOutputError(f"model returned no valid {schema.__name__}: {error}")
