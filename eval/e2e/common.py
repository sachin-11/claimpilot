"""Paths, config and case loading for the end-to-end evaluation."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.engine import make_url

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent  # "/" inside the backend container (eval is mounted at /eval)
DATA_DIR = REPO_ROOT / "data"
CONFIGS_DIR = EVAL_DIR / "configs"
RESULTS_DIR = EVAL_DIR / "results"
ADVERSARIAL_DIR = EVAL_DIR / "adversarial"
WORK_DIR = EVAL_DIR / "work"  # gitignored: the data root the eval database is seeded from
THRESHOLDS_PATH = EVAL_DIR / "thresholds.yaml"
RULE_MAP_PATH = EVAL_DIR / "rule_map.yaml"
PRICING_PATH = EVAL_DIR.parent / "rag" / "pricing.yaml"

AMOUNT_TOLERANCE = 1.0  # rupees
RUN_PREFIX = "eval-"


class UnsafeDatabaseError(RuntimeError):
    """The evaluation was pointed at a database that is not clearly an eval database."""


def assert_eval_database(eval_url: str, main_url: str) -> None:
    """Refuse to run against the main database: the eval deletes decisions and resets claims."""
    ev, main = make_url(eval_url), make_url(main_url)
    same_server = (ev.host, ev.port) == (main.host, main.port)
    if same_server and ev.database == main.database:
        raise UnsafeDatabaseError("EVAL_DATABASE_URL is the main DATABASE_URL: refusing to run")
    if "eval" not in (ev.database or "").lower():
        raise UnsafeDatabaseError(
            f"database name {ev.database!r} does not contain 'eval': refusing to run"
        )


def default_eval_url(main_url: str) -> str:
    main = make_url(main_url)
    return main.set(database=f"{main.database}_eval").render_as_string(hide_password=False)


def eval_database_url(main_url: str) -> str:
    return os.environ.get("EVAL_DATABASE_URL") or default_eval_url(main_url)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@dataclass
class EvalConfig:
    name: str
    description: str = ""
    supervisor_llm: bool = True
    model: str = "default"  # default | groq_only
    reranker: str = "none"
    fresh_extraction: bool = True
    synthetic_wording: bool = False  # ingest generated wordings for the policies without a PDF


def load_config(name: str) -> EvalConfig:
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(
            f"unknown config {name!r}; available: "
            f"{sorted(p.stem for p in CONFIGS_DIR.glob('*.yaml'))}"
        )
    return EvalConfig(**load_yaml(path))


@dataclass
class Case:
    claim_id: str
    kind: str  # golden | adversarial
    golden: dict[str, Any]
    truth: dict[str, Any] | None = None
    adversarial: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario(self) -> str:
        return self.golden.get("scenario_type", "adversarial")


def load_cases(
    *,
    data_dir: Path = DATA_DIR,
    adversarial_dir: Path = ADVERSARIAL_DIR,
    include_adversarial: bool = True,
) -> list[Case]:
    expected = json.loads((data_dir / "golden" / "expected.json").read_text(encoding="utf-8"))
    truth_rows = json.loads(
        (data_dir / "golden" / "extraction_truth.json").read_text(encoding="utf-8")
    )
    truth = {row["claim_id"]: row for row in truth_rows}
    cases = [Case(r["claim_id"], "golden", r, truth.get(r["claim_id"])) for r in expected]
    adv_path = adversarial_dir / "expected.json"
    if include_adversarial and adv_path.exists():
        for rec in json.loads(adv_path.read_text(encoding="utf-8")):
            cases.append(
                Case(rec["claim_id"], "adversarial", {"policy_id": rec["policy_id"]}, None, rec)
            )
    return cases


def load_pricing() -> dict[str, dict[str, float]]:
    return load_yaml(PRICING_PATH).get("models", {})


def price_usd(
    model: str, tokens_in: int, tokens_out: int, pricing: dict[str, dict[str, float]]
) -> float:
    """Cost estimate; the model is matched by its longest name prefix."""
    best = max((m for m in pricing if model.startswith(m)), key=len, default=None)
    if best is None:
        return 0.0
    p = pricing[best]
    return (tokens_in * p["input_per_million"] + tokens_out * p["output_per_million"]) / 1_000_000
