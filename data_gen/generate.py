"""CLI: python -m data_gen.generate --count 40 --seed 42 --out data"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Annotated

import typer

from data_gen.config import load_policy_rules, load_typical_costs
from data_gen.llm_text import DEFAULT_MODEL, load_env_file
from data_gen.planner import DEFAULT_MIX, plan_claims
from data_gen.writer import clean_previous_output, write_claim, write_golden

logger = logging.getLogger("data_gen")
app = typer.Typer(add_completion=False, help="Generate synthetic ClaimPilot claims + golden dataset.")


def _parse_mix(text: str | None) -> dict[str, int] | None:
    if not text:
        return None
    mix: dict[str, int] = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in DEFAULT_MIX:
            raise typer.BadParameter(f"Unknown scenario '{key}'. Use: {', '.join(DEFAULT_MIX)}")
        mix[key] = int(value)
    return mix


@app.command()
def main(
    count: Annotated[int, typer.Option(help="Total claims (scaled from the default 15/8/7/6/4 mix)")] = 40,
    seed: Annotated[int, typer.Option(help="Same seed = same output (except cached LLM text)")] = 42,
    out: Annotated[Path, typer.Option(help="Output root containing claims/ and golden/")] = Path("data"),
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Force template narratives")] = False,
    mix: Annotated[
        str | None,
        typer.Option(help="Explicit mix, e.g. covered=15,partial=8,rejected=7,risk=6,incomplete=4 (overrides --count)"),
    ] = None,
    scan_count: Annotated[int, typer.Option(help="Claims that also get *_scanned.png copies")] = 10,
    policy_rules: Annotated[Path | None, typer.Option(help="Override policy_rules.yaml")] = None,
    typical_costs: Annotated[Path | None, typer.Option(help="Override typical_costs.yaml")] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env_file()
    model = os.getenv("GEN_MODEL", DEFAULT_MODEL)
    use_llm = not no_llm and bool(os.getenv("OPENAI_API_KEY"))
    logger.info("Narratives: %s", f"OpenAI {model} (cached)" if use_llm else "templates")

    rules = load_policy_rules(policy_rules)
    costs = load_typical_costs(typical_costs)
    planned = plan_claims(count, seed, rules, costs, _parse_mix(mix))

    scan_ids = set(random.Random(f"{seed}:scan").sample([p.spec.claim_id for p in planned], min(scan_count, len(planned))))
    for p in planned:
        p.golden.has_scanned_copy = p.spec.claim_id in scan_ids

    clean_previous_output(out)
    for p in planned:
        write_claim(p, out, use_llm=use_llm, model=model, seed=seed, scan=p.spec.claim_id in scan_ids)
    write_golden([p.golden for p in planned], out)
    logger.info("Wrote %d claims (%d scanned) to %s", len(planned), len(scan_ids), out.resolve())


if __name__ == "__main__":
    app()
