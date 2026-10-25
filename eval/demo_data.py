"""`python -m eval.demo_data`: demo and UI test data that needs no API key and no real policy PDF.

1. writes a synthetic policy wording PDF per policy (from `policy_rules.yaml`, the same generator
   the evaluation uses) into `data/policies_demo/`, never over the real wording in `data/policies/`;
2. ingests them (`POLICIES_DIR` points the ingestion at that folder): with `MOCK_LLM=true` the
   embeddings are the deterministic fake ones, otherwise the real embedding model is called;
3. creates demo users (idempotent): two officers and an admin, all with DEMO_PASSWORD.

Run `make seed` first (policies, customers and claims must exist); `make demo` does all of it.
"""

import asyncio
import os
import subprocess
import sys

import bcrypt
import yaml
from sqlalchemy import text

from app.core.config import get_settings
from app.core.db import create_engine
from eval.e2e.synthetic_wording import write_wordings

DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "demo-pass-12345")
DEMO_USERS = (
    ("demo-officer@example.test", "Demo Officer", "officer"),
    ("demo-officer2@example.test", "Second Officer", "officer"),
    ("demo-admin@example.test", "Demo Admin", "admin"),
)


def write_demo_wordings() -> list[str]:
    settings = get_settings()
    rules = yaml.safe_load(settings.policy_rules_path.read_text(encoding="utf-8"))
    target = settings.data_dir / "policies_demo"
    paths = write_wordings(rules["policies"], target)
    sys.stdout.write(f"wrote {len(paths)} synthetic wordings to {target}\n")
    return list(paths)


def ingest(code: str) -> None:
    folder = get_settings().data_dir / "policies_demo"
    env = {**os.environ, "POLICIES_DIR": str(folder)}
    result = subprocess.run(
        [sys.executable, "-m", "app.ingestion.run", "--policy", code, "--force"],
        env=env,
        capture_output=True,
        text=True,
    )
    last = (result.stdout.strip().splitlines() or [""])[-1]
    sys.stdout.write(f"ingest {code}: {last[:160] or result.stderr.strip()[-200:]}\n")
    if result.returncode != 0:
        raise SystemExit(
            f"ingestion of {code} failed:\n{result.stdout[-600:]}\n{result.stderr[-600:]}"
        )


async def upsert_users() -> None:
    password_hash = bcrypt.hashpw(DEMO_PASSWORD.encode()[:72], bcrypt.gensalt(rounds=10)).decode()
    engine = create_engine(get_settings().database_url)
    async with engine.begin() as conn:
        for email, name, role in DEMO_USERS:
            await conn.execute(
                text(
                    "INSERT INTO users (email, full_name, role, password_hash) "
                    "VALUES (:e, :n, CAST(:r AS user_role), :h) "
                    "ON CONFLICT (email) DO UPDATE SET password_hash = :h, is_active = true"
                ),
                {"e": email, "n": name, "r": role, "h": password_hash},
            )
    await engine.dispose()
    sys.stdout.write("demo users: " + ", ".join(email for email, _, _ in DEMO_USERS) + "\n")


def main() -> None:
    for code in write_demo_wordings():
        ingest(code)
    asyncio.run(upsert_users())
    sys.stdout.write(f"password for the demo users: {DEMO_PASSWORD}\n")


if __name__ == "__main__":
    main()
