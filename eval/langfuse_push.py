"""Push evaluation results to Langfuse (opt-in: `--langfuse` on the e2e and RAG eval runners).

What shows up in the Langfuse UI:
- a dataset per golden set (`claimpilot-golden-e2e`, `claimpilot-golden-rag`), one item per case
  (upserted by the case id, so it is safe to run again);
- one dataset run per evaluation (`<config>-<timestamp>`), linking each case's trace to its item,
  so runs can be compared side by side;
- scores on each trace: decision_correct, amount_correct, citation_correct (e2e), faithfulness,
  hit_at_3, citation_correct (rag), adversarial_pass.

The traces themselves are created by the app's own tracing (`scenario:eval`); this module only adds
datasets, run links and scores. Without Langfuse keys everything here is a no-op."""

import logging
from typing import Any

import httpx

from app.core.config import get_settings, has_real_key
from app.observability import tracing

logger = logging.getLogger("eval.langfuse")

E2E_DATASET = "claimpilot-golden-e2e"
RAG_DATASET = "claimpilot-golden-rag"


def init() -> bool:
    """Start tracing from the environment; False when no Langfuse keys are set."""
    return tracing.init_tracing(get_settings())


def ready() -> bool:
    return tracing.tracing_enabled()


def _client() -> Any:
    return tracing._client  # noqa: SLF001 - the one Langfuse client of the process


def ensure_dataset(name: str, description: str, items: list[dict[str, Any]]) -> None:
    """Create the dataset if needed and upsert its items (id, input, expected_output, metadata)."""
    client = _client()
    if client is None:
        return
    try:
        client.create_dataset(name=name, description=description)
    except Exception:  # noqa: BLE001 - it already exists
        logger.debug("dataset_exists", extra={"dataset": name})
    for item in items:
        try:
            client.create_dataset_item(dataset_name=name, **item)
        except Exception:  # noqa: BLE001
            logger.warning("dataset_item_failed", extra={"item": item.get("id")}, exc_info=True)


def link_run(run_name: str, item_id: str, trace_id: str, metadata: Any = None) -> None:
    """Attach a trace to a dataset item (created with our id) as part of the run `run_name`.
    REST call: the SDK has no helper for it."""
    settings = get_settings()
    if not (
        has_real_key(settings.langfuse_public_key) and has_real_key(settings.langfuse_secret_key)
    ):
        return
    base = (
        settings.langfuse_base_url or settings.langfuse_host or "https://cloud.langfuse.com"
    ).rstrip("/")
    try:
        response = httpx.post(
            f"{base}/api/public/dataset-run-items",
            auth=(
                settings.langfuse_public_key.get_secret_value(),  # type: ignore[union-attr]
                settings.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
            ),
            json={
                "runName": run_name,
                "datasetItemId": item_id,
                "traceId": trace_id,
                "metadata": metadata,
            },
            timeout=20,
        )
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        logger.warning("dataset_run_link_failed", extra={"item": item_id}, exc_info=True)


def push_scores(
    trace_id: str, scores: dict[str, float | bool | None], comment: str | None = None
) -> None:
    for name, value in scores.items():
        if value is None:
            continue
        tracing.score_trace(trace_id, name, float(value), comment)


def finish() -> None:
    tracing.flush_tracing()
