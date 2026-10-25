"""Read-only access to the evaluation results the two runners write under `eval/<kind>/results/`.

Only `report.md` and the small summary JSON files are exposed (never `runs.jsonl`, which holds the
full agent state). A result is addressed by `<kind>:<folder>`; both parts are validated against a
strict pattern AND the resolved path has to stay inside the results folder, so no id can reach
another file."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

KINDS = ("rag", "e2e")
FOLDER_RE = re.compile(r"^\d{8}-\d{6}_[A-Za-z0-9_.+-]{1,100}$")
MAX_REPORT_BYTES = 2 * 1024 * 1024
HEADLINE_E2E = (
    "decision_accuracy",
    "amount_within_tolerance",
    "adversarial_pass_rate",
    "error_rate",
    "cost_per_claim_usd",
    "latency_p95_s",
)


class EvalNotFoundError(LookupError):
    pass


def _results_dir(eval_dir: Path, kind: str) -> Path:
    return eval_dir / kind / "results"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _created(folder: str) -> datetime | None:
    try:
        return datetime.strptime(folder[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def _headline(kind: str, folder: Path) -> dict[str, Any]:
    """A handful of numbers for the list, whatever the runner wrote."""
    if kind == "e2e":
        metrics = (_json(folder / "metrics.json") or {}).get("all") or {}
        out = {k: metrics.get(k) for k in HEADLINE_E2E if k in metrics}
        out["claims"] = metrics.get("claims")
        return out
    summary = _json(folder / "summary.json") or {}
    first = next(iter((summary.get("configs") or {}).values()), {})
    metrics = first.get("metrics") or {}
    return {
        k: metrics.get(k)
        for k in ("hit_at_3", "mrr", "faithfulness", "citation_ok")
        if k in metrics
    }


def list_eval_runs(eval_dir: Path, limit: int = 200) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for kind in KINDS:
        base = _results_dir(eval_dir, kind)
        if not base.is_dir():
            continue
        for folder in base.iterdir():
            if not folder.is_dir() or not FOLDER_RE.match(folder.name):
                continue
            created = _created(folder.name)
            runs.append(
                {
                    "id": f"{kind}:{folder.name}",
                    "kind": kind,
                    "name": folder.name.split("_", 1)[1],
                    "created_at": created.isoformat() if created else None,
                    "has_report": (folder / "report.md").is_file(),
                    "headline": _headline(kind, folder),
                }
            )
    runs.sort(key=lambda r: r["created_at"] or "", reverse=True)
    return runs[:limit]


def read_eval_run(eval_dir: Path, eval_id: str) -> dict[str, Any]:
    kind, _, folder_name = eval_id.partition(":")
    if kind not in KINDS or not FOLDER_RE.match(folder_name):
        raise EvalNotFoundError(eval_id)
    base = _results_dir(eval_dir, kind).resolve()
    folder = (base / folder_name).resolve()
    if not folder.is_relative_to(base) or not folder.is_dir():
        raise EvalNotFoundError(eval_id)
    report = folder / "report.md"
    text = ""
    if report.is_file():
        text = report.read_bytes()[:MAX_REPORT_BYTES].decode("utf-8", errors="replace")
    created = _created(folder_name)
    return {
        "id": eval_id,
        "kind": kind,
        "name": folder_name.split("_", 1)[1],
        "created_at": created.isoformat() if created else None,
        "report_md": text,
        "headline": _headline(kind, folder),
        "config": _json(folder / "config.json") if kind == "e2e" else None,
    }
