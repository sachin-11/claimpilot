"""The isolated evaluation database: create, migrate, seed, ingest policies.

Nothing here ever opens the main DATABASE_URL for writing; `assert_eval_database` is checked first.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from eval.e2e.common import (
    ADVERSARIAL_DIR,
    DATA_DIR,
    RULE_MAP_PATH,
    WORK_DIR,
    assert_eval_database,
    load_yaml,
)
from eval.e2e.synthetic_wording import write_wordings

logger = logging.getLogger("eval.e2e.setup")
BACKEND_DIR = (
    Path("/app")
    if Path("/app/alembic.ini").exists()
    else Path(__file__).resolve().parents[2] / "backend"
)


async def create_database(eval_url: str) -> None:
    url = make_url(eval_url)
    admin = create_async_engine(
        url.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool
    )
    async with admin.connect() as conn:
        exists = (
            await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            )
        ).scalar()
        if not exists:
            await conn.execute(text(f'CREATE DATABASE "{url.database}"'))
            logger.info("created database %s", url.database)
    await admin.dispose()


def migrate(eval_url: str) -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.attributes["database_url"] = eval_url
    command.upgrade(cfg, "head")


def build_work_dir(
    data_dir: Path = DATA_DIR, adversarial_dir: Path = ADVERSARIAL_DIR, work: Path = WORK_DIR
) -> Path:
    """One data root with the golden claims, the adversarial claims and the policy PDFs, so the
    seed script, the intake service and the claim-form reader resolve the same relative paths."""
    root = work / "data"
    (root / "claims").mkdir(parents=True, exist_ok=True)
    for source in (data_dir / "claims", adversarial_dir / "claims"):
        if source.exists():
            for claim in source.iterdir():
                if claim.is_dir():
                    shutil.copytree(claim, root / "claims" / claim.name, dirs_exist_ok=True)
    if (data_dir / "policies").exists():
        shutil.copytree(data_dir / "policies", root / "policies", dirs_exist_ok=True)
    return root


def _run_module(module: str, args: list[str], eval_url: str, data_root: Path) -> None:
    env = {**os.environ, "DATABASE_URL": eval_url, "DATA_DIR": str(data_root)}
    proc = subprocess.run(
        [sys.executable, "-m", module, *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=BACKEND_DIR,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{module} failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-800:]}"
        )
    logger.info("%s ok", module)


async def setup(eval_url: str, main_url: str) -> Path:
    assert_eval_database(eval_url, main_url)
    await create_database(eval_url)
    await asyncio.to_thread(migrate, eval_url)
    data_root = await asyncio.to_thread(build_work_dir)
    await asyncio.to_thread(_run_module, "app.scripts.seed", [], eval_url, data_root)
    # only policies whose PDF exists can be ingested (`--all` fails on the others)
    for code in load_yaml(RULE_MAP_PATH).get("wording_policies") or []:
        await asyncio.to_thread(
            _run_module, "app.ingestion.run", ["--policy", code], eval_url, data_root
        )
    return data_root


async def sync_wordings(
    engine: AsyncEngine, eval_url: str, data_root: Path, desired: list[str], synthetic: list[str]
) -> None:
    """Make the ingested wordings exactly `desired`: synthetic ones are written, ingested if absent,
    and removed again when a config does not want them (so configs never leak into each other)."""
    from app.core.config import get_settings  # noqa: PLC0415 - settings need the env of the container

    rules = load_yaml(get_settings().policy_rules_path)
    by_code = {p["policy_id"]: p for p in rules["policies"]}
    wanted_synthetic = [by_code[c] for c in synthetic if c in desired]
    await asyncio.to_thread(write_wordings, wanted_synthetic, data_root / "policies")
    async with engine.begin() as conn:
        for code in synthetic:  # always refreshed: the generated text may have changed
            for table in ("policy_chunks", "ingestion_runs"):
                await conn.execute(
                    text(
                        f"DELETE FROM {table} WHERE policy_id = "
                        "(SELECT id FROM policies WHERE policy_code = :code)"
                    ),
                    {"code": code},
                )
        have = {
            row[0]
            for row in await conn.execute(
                text(
                    "SELECT p.policy_code FROM policies p "
                    "WHERE EXISTS (SELECT 1 FROM policy_chunks c WHERE c.policy_id = p.id)"
                )
            )
        }
    for code in desired:
        if code in synthetic or code not in have:
            await asyncio.to_thread(
                _run_module, "app.ingestion.run", ["--policy", code, "--force"], eval_url, data_root
            )


RESET_SQL = [
    "DELETE FROM claim_approvals",
    "DELETE FROM notifications",
    "DELETE FROM tool_idempotency",
    "DELETE FROM claim_decisions",
    "DELETE FROM checkpoint_writes WHERE thread_id LIKE 'eval-%'",
    "DELETE FROM checkpoint_blobs WHERE thread_id LIKE 'eval-%'",
    "DELETE FROM checkpoints WHERE thread_id LIKE 'eval-%'",
    "UPDATE claims SET status = 'submitted', approved_amount = NULL",
]


async def reset_state(engine: AsyncEngine, *, drop_extractions: bool) -> None:
    """Back to 'every claim is new' between passes (policies and users are kept)."""
    statements = list(RESET_SQL)
    if drop_extractions:
        statements.append("DELETE FROM claim_extractions")
    for statement in statements:
        try:
            async with (
                engine.begin() as conn
            ):  # one transaction each: a failure must not poison the rest
                await conn.execute(text(statement))
        except Exception:  # noqa: BLE001 - the checkpoint tables do not exist before the first run
            if "checkpoint" not in statement:
                raise
