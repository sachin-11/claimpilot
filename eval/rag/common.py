"""Shared helpers for the RAG evaluation: paths, jsonl IO, text normalisation."""

import json
import re
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = EVAL_DIR / "golden"
QUESTIONS_PATH = GOLDEN_DIR / "questions.jsonl"  # human-verified questions only
CANDIDATES_PATH = GOLDEN_DIR / "candidates.jsonl"  # LLM proposals, never trusted
SKIPPED_PATH = GOLDEN_DIR / "skipped_candidates.json"
CONFIGS_DIR = EVAL_DIR / "configs"
RESULTS_DIR = EVAL_DIR / "results"
THRESHOLDS_PATH = EVAL_DIR / "thresholds.yaml"


def normalize_text(text: str) -> str:
    """Fold case / spacing so "24 Months" == "24 months" and "5 %" == "5%".

    Also: thousands separators ("10,000" == "10000"), "Rs. 5,000/-" == "rs.5000", dashes and
    quotes. Digits, units and amounts are never altered beyond that.
    """
    text = text.lower().replace(" ", " ")
    text = text.replace("–", "-").replace("—", "-").replace("’", "'")
    text = text.replace("₹", "rs.")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 10,000 -> 10000
    text = re.sub(r"\s*%", "%", text)  # 5 % -> 5%
    text = re.sub(r"\brs\.?\s*", "rs.", text)  # Rs. 5000 / Rs 5000 -> rs.5000
    text = re.sub(r"/-", "", text)  # 5000/- -> 5000
    return " ".join(text.split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
