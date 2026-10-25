"""Admin dashboard data: evaluation results on disk (no path can escape) and the metrics summary."""

import json
import os
from pathlib import Path

import pytest
from agent_helpers import make_agent_world
from auth_helpers import bearer
from fake_redis import FakeRedis
from sqlalchemy.ext.asyncio import AsyncSession
from test_auth import client_for, make_settings, make_user

from app.core.eval_results import EvalNotFoundError, list_eval_runs, read_eval_run
from app.models import UserRole
from app.observability.usage import runs_key, spend_key, today


def write_results(root: Path) -> None:
    e2e = root / "e2e" / "results"
    default = e2e / "20261025-154911_default"
    default.mkdir(parents=True)
    (default / "report.md").write_text("# End-to-end `default`\n\nAccuracy 27.5%")
    (default / "metrics.json").write_text(
        json.dumps({"all": {"decision_accuracy": 0.275, "claims": 40, "cost_per_claim_usd": 0.006}})
    )
    (default / "config.json").write_text(json.dumps({"name": "default"}))
    (default / "runs.jsonl").write_text(json.dumps({"secret": "full agent state"}) + "\n")
    (e2e / "20261025-154911_compare").mkdir()
    (e2e / "20261025-154911_compare" / "report.md").write_text("# Configuration comparison")
    (e2e / "not-a-run-folder").mkdir()  # ignored: the name is not <date>_<name>
    rag = root / "rag" / "results" / "20261020-101500_hybrid_rerank"
    rag.mkdir(parents=True)
    (rag / "report.md").write_text("# RAG evaluation")
    (rag / "summary.json").write_text(
        json.dumps({"configs": {"hybrid": {"metrics": {"hit_at_3": 0.9}}}})
    )


def test_runs_are_listed_newest_first_with_headline_numbers(tmp_path: Path) -> None:
    write_results(tmp_path)
    runs = list_eval_runs(tmp_path)
    assert len(runs) == 3 and runs[-1]["id"] == "rag:20261020-101500_hybrid_rerank"  # the oldest
    by_id = {r["id"]: r for r in runs}
    default = by_id["e2e:20261025-154911_default"]
    assert default["kind"] == "e2e" and default["name"] == "default" and default["has_report"]
    assert default["headline"]["decision_accuracy"] == 0.275 and default["headline"]["claims"] == 40
    assert by_id["rag:20261020-101500_hybrid_rerank"]["headline"] == {"hit_at_3": 0.9}
    assert by_id["e2e:20261025-154911_compare"]["headline"]["claims"] is None
    assert all("not-a-run" not in r["id"] for r in runs)
    assert list_eval_runs(tmp_path / "missing") == []


def test_one_run_returns_the_report_but_never_the_run_log(tmp_path: Path) -> None:
    write_results(tmp_path)
    run = read_eval_run(tmp_path, "e2e:20261025-154911_default")
    assert run["report_md"].startswith("# End-to-end") and run["config"] == {"name": "default"}
    assert "secret" not in json.dumps(run)  # runs.jsonl is not exposed


@pytest.mark.parametrize(
    "bad",
    [
        "e2e:../../etc",
        "e2e:20261025-154911_default/../../x",
        "../rag:20261020-101500_hybrid_rerank",
        "e2e:20261025-154911_default/report.md",
        "etc:passwd",
        "e2e:",
        ":20261025-154911_default",
        "e2e:20261025-154911_" + "a" * 200,
        "e2e:20261025-154911_default\x00",
        "20261025-154911_default",
    ],
)
def test_ids_cannot_escape_the_results_folder(tmp_path: Path, bad: str) -> None:
    write_results(tmp_path)
    with pytest.raises(EvalNotFoundError):
        read_eval_run(tmp_path, bad)


def test_a_symlinked_folder_pointing_outside_is_not_followed(tmp_path: Path) -> None:
    write_results(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "report.md").write_text("private")
    link = tmp_path / "e2e" / "results" / "20261001-000000_link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available here")
    with pytest.raises(EvalNotFoundError):
        read_eval_run(tmp_path, "e2e:20261001-000000_link")


async def test_eval_endpoints_are_admin_only_and_404_on_unknown_ids(
    session: AsyncSession, tmp_path: Path
) -> None:
    write_results(tmp_path)
    officer = await make_user(session, "off@example.com")
    admin = await make_user(session, "boss@example.com", UserRole.ADMIN)
    settings = make_settings(eval_dir=tmp_path)
    async with client_for(session, settings) as (http, _, _):
        as_officer = await http.get("/admin/evals", headers=bearer(settings, officer))
        assert as_officer.status_code == 403
        headers = bearer(settings, admin)
        listing = await http.get("/admin/evals", headers=headers)
        assert listing.status_code == 200 and len(listing.json()) == 3
        one = await http.get("/admin/evals/e2e:20261025-154911_default", headers=headers)
        assert one.status_code == 200 and "Accuracy 27.5%" in one.json()["report_md"]
        encoded = await http.get("/admin/evals/e2e%3A20261025-154911_default", headers=headers)
        assert encoded.status_code == 200
        for bad in ("e2e:..%2F..%2Fetc", "rag:nope", "x"):
            assert (await http.get(f"/admin/evals/{bad}", headers=headers)).status_code == 404
        assert (await http.get("/admin/evals")).status_code == 401


async def test_metrics_summary_counts_spend_fallback_runs_and_claims(session: AsyncSession) -> None:
    officer = await make_user(session, "off@example.com")
    admin = await make_user(session, "boss@example.com", UserRole.ADMIN)
    await make_agent_world(session)  # one claim, status submitted
    settings = make_settings(daily_llm_budget_inr=200.0)
    redis = FakeRedis()
    redis.hashes[spend_key(today())] = {
        "inr": "50.0",
        "calls": "10",
        "tokens_in": "1000",
        "tokens_out": "100",
        "calls:openai": "9",
        "calls:groq": "1",
        "inr:openai:gpt-4o-mini": "50.0",
    }
    redis.hashes[runs_key(today())] = {"awaiting_approval": "3", "failed": "1"}
    async with client_for(session, settings, redis=redis) as (http, _, _):
        forbidden = await http.get("/admin/metrics/summary", headers=bearer(settings, officer))
        assert forbidden.status_code == 403
        body = (await http.get("/admin/metrics/summary", headers=bearer(settings, admin))).json()
    assert body["spend_inr"] == 50.0 and body["budget_inr"] == 200.0
    assert body["budget_used_percent"] == 25.0
    assert body["llm_calls"] == 10 and body["fallback_calls"] == 1 and body["fallback_rate"] == 0.1
    assert body["calls_by_provider"] == {"openai": 9, "groq": 1}
    assert body["runs_today"] == {"awaiting_approval": 3, "failed": 1}
    assert body["claims_by_status"]["submitted"] == 1 and body["claims_by_status"]["approved"] == 0
    assert set(body["claims_by_status"]) >= {
        "submitted",
        "processing",
        "awaiting_approval",
        "rejected",
    }


async def test_metrics_summary_with_nothing_recorded_yet(session: AsyncSession) -> None:
    admin = await make_user(session, "boss@example.com", UserRole.ADMIN)
    settings = make_settings(daily_llm_budget_inr=0)
    async with client_for(session, settings) as (http, _, _):
        body = (await http.get("/admin/metrics/summary", headers=bearer(settings, admin))).json()
    assert body["fallback_rate"] is None and body["budget_used_percent"] is None
    assert body["spend_inr"] == 0 and body["runs_today"] == {} and body["llm_calls"] == 0
