"""LLM list prices (USD per 1M tokens) for cost estimates: the metrics, the budget guard and the
evaluation reports all read the same table, `pricing.yaml` next to this file. The prices are
ASSUMPTIONS from public price pages: edit the file when they change."""

from functools import lru_cache
from pathlib import Path

import yaml

PRICING_PATH = Path(__file__).with_name("pricing.yaml")


@lru_cache(maxsize=1)
def load_pricing() -> dict[str, dict[str, float]]:
    data = yaml.safe_load(PRICING_PATH.read_text(encoding="utf-8")) or {}
    return data.get("models", {})


def cost_usd(
    model: str,
    tokens_in: int,
    tokens_out: int,
    pricing: dict[str, dict[str, float]] | None = None,
) -> float:
    """USD for one call; the model is matched by its longest name prefix
    ("gpt-4o-mini-2024-07-18" -> "gpt-4o-mini"). An unknown model costs 0."""
    table = pricing if pricing is not None else load_pricing()
    best = max((m for m in table if model.startswith(m)), key=len, default=None)
    if best is None:
        return 0.0
    p = table[best]
    return (tokens_in * p["input_per_million"] + tokens_out * p["output_per_million"]) / 1_000_000
