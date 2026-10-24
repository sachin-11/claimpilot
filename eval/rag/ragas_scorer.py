"""RAGAS scoring worker. Runs INSIDE the isolated virtualenv (/opt/ragas-venv), not the app env.

    /opt/ragas-venv/bin/python -m eval.rag.ragas_scorer in.jsonl out.jsonl

Why a separate environment: ragas 0.4.3 pins an older LangChain stack that would downgrade the
app's langchain-openai. The app talks to this worker through plain JSONL files.

Input rows : {id, user_input, response, retrieved_contexts: [str], reference}
Output rows: {id, faithfulness, answer_relevancy, context_precision, context_recall, errors}
Last line of the output file is {"_summary": {...judge tokens...}}.
Environment: OPENAI_API_KEY, EVAL_JUDGE_MODEL (default gpt-4o), EVAL_EMBEDDING_MODEL.
This file deliberately imports nothing from the `app` package.
"""

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path

import httpx
from openai import AsyncOpenAI

warnings.filterwarnings("ignore")

from ragas.embeddings import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

CONCURRENCY = int(os.environ.get("EVAL_JUDGE_CONCURRENCY", "4"))


class TokenCounter:
    """Counts judge tokens from the `usage` block of every OpenAI response."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_tokens = 0
        self.calls = 0

    async def on_response(self, response: httpx.Response) -> None:
        await response.aread()
        try:
            usage = response.json().get("usage") or {}
        except ValueError:
            return
        self.calls += 1
        if str(response.request.url).endswith("/embeddings"):
            self.embedding_tokens += usage.get("total_tokens", usage.get("prompt_tokens", 0))
        else:
            self.input_tokens += usage.get("prompt_tokens", 0)
            self.output_tokens += usage.get("completion_tokens", 0)


def build_metrics(client: AsyncOpenAI, judge_model: str, embedding_model: str) -> dict:
    llm = llm_factory(judge_model, provider="openai", client=client, temperature=0)
    embeddings = embedding_factory("openai", model=embedding_model, client=client)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_row(metrics: dict, row: dict, semaphore: asyncio.Semaphore) -> dict:
    question, response = row["user_input"], row["response"]
    contexts, reference = row["retrieved_contexts"], row["reference"]
    calls = {
        "faithfulness": lambda: metrics["faithfulness"].ascore(
            user_input=question, response=response, retrieved_contexts=contexts
        ),
        "answer_relevancy": lambda: metrics["answer_relevancy"].ascore(
            user_input=question, response=response
        ),
        "context_precision": lambda: metrics["context_precision"].ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        ),
        "context_recall": lambda: metrics["context_recall"].ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference
        ),
    }
    result: dict = {"id": row["id"], "errors": {}}

    async def run(name: str, call) -> None:
        async with semaphore:
            try:
                value = (await call()).value
                result[name] = None if value is None else float(value)
            except Exception as exc:  # noqa: BLE001 - one metric failing must not lose the rest
                result[name] = None
                result["errors"][name] = f"{type(exc).__name__}: {str(exc)[:200]}"

    await asyncio.gather(*(run(name, call) for name, call in calls.items()))
    return result


async def main(input_path: Path, output_path: Path) -> None:
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o")
    embedding_model = os.environ.get("EVAL_EMBEDDING_MODEL", "text-embedding-3-small")
    counter = TokenCounter()
    http_client = httpx.AsyncClient(
        event_hooks={"response": [counter.on_response]}, timeout=httpx.Timeout(120.0)
    )
    client = AsyncOpenAI(http_client=http_client)
    metrics = build_metrics(client, judge_model, embedding_model)

    text = await asyncio.to_thread(input_path.read_text, encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line]
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(score_row(metrics, row, semaphore) for row in rows))

    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        summary = {
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "judge_input_tokens": counter.input_tokens,
            "judge_output_tokens": counter.output_tokens,
            "embedding_tokens": counter.embedding_tokens,
            "api_calls": counter.calls,
        }
        handle.write(json.dumps({"_summary": summary}) + "\n")
    await http_client.aclose()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python -m eval.rag.ragas_scorer in.jsonl out.jsonl\n")
        raise SystemExit(2)
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
