.PHONY: e2e-up e2e-down e2e demo-data auditlangfuse-up langfuse-down mcp-smoke up down logs test lint fmt migrate seed ingest ingest-dry try-queries eval-candidates eval-review eval-rag eval-run eval-e2e revision shell

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f backend

# Database tests need postgres (they use a separate "<db>_test" database).
test:
	docker compose run --rm backend pytest

audit:
	docker compose run --rm --no-deps backend pip-audit --skip-editable

lint:
	docker compose run --rm --no-deps backend ruff check --config pyproject.toml . ../eval ../mcp_servers

fmt:
	docker compose run --rm --no-deps backend sh -c "ruff format --config pyproject.toml . ../eval ../mcp_servers && ruff check --config pyproject.toml --fix . ../eval ../mcp_servers"

migrate:
	docker compose exec backend alembic upgrade head

mcp-smoke:
	docker compose exec backend python -m app.mcp_client.smoke

seed:
	docker compose exec backend python -m app.scripts.seed

ingest:
	docker compose exec backend python -m app.ingestion.run --all

ingest-dry:
	docker compose exec backend python -m app.ingestion.run --all --dry-run

# Demo without API cost: MOCK_LLM=true make up, then make seed demo-data. Writes synthetic wordings to
# data/policies_demo/, ingests them and creates demo users (demo-officer@example.test, demo-admin@example.test).
demo-data:
	docker compose exec -T backend python -m eval.demo_data

# manual retrieval check against real data (needs `make ingest` first); pass ARGS="--reranker local"
try-queries:
	docker compose exec backend python -m app.rag.try_queries $(ARGS)

# Golden set for RAG eval: generate candidates, then review them (interactive terminal)
eval-candidates:
	docker compose exec backend python -m eval.rag.build_candidates $(ARGS)

eval-review:
	docker compose exec backend python -m eval.rag.review $(ARGS)

# RAG regression gate: default config vs eval/rag/thresholds.yaml (exit 1 if below)
eval-rag:
	docker compose exec -T backend python -m eval.rag.run_eval --gate

# usage: make eval-run ARGS="--configs hybrid_no_rerank hybrid_rerank_local --no-ragas"
eval-run:
	docker compose exec -T backend python -m eval.rag.run_eval $(ARGS)

# usage: make revision m="add claims table"
revision:
	docker compose exec backend alembic revision --autogenerate -m "$(m)"

shell:
	docker compose exec backend bash

# End-to-end agent evaluation (Module 9): default config vs eval/e2e/thresholds.yaml (exit 1 if below).
# Uses the separate eval database. The gate is checked on the FIRST config: synthetic_wording (every
# policy has a wording), because `default` only has the real POL-AS wording and cannot reach the
# thresholds by design. Usage: make eval-e2e ARGS="--configs synthetic_wording deterministic_routing --langfuse"
eval-e2e:
	docker compose exec -T backend python -m eval.e2e.run_eval --gate --configs synthetic_wording default $(ARGS)

# Self-hosted Langfuse (opt-in, ~4 GB RAM): UI http://localhost:3001. Put LANGFUSE_PUBLIC_KEY,
# LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL=http://langfuse-web:3000 in .env (see .env.example).
langfuse-up:
	docker compose -f docker-compose.yml -f docker-compose.langfuse.yml --profile observability up -d langfuse-web langfuse-worker

langfuse-down:
	docker compose -f docker-compose.yml -f docker-compose.langfuse.yml --profile observability stop langfuse-web langfuse-worker langfuse-clickhouse langfuse-minio langfuse-redis langfuse-postgres

# --- Browser end-to-end tests (Module 11): an isolated stack (own DB, MOCK_LLM=true, no API cost) ---
E2E_COMPOSE = MOCK_LLM=true docker compose -p claimpilot-e2e -f docker-compose.yml -f docker-compose.e2e.yml

e2e-up:
	$(E2E_COMPOSE) up -d --build --wait
	$(E2E_COMPOSE) exec -T backend alembic upgrade head
	$(E2E_COMPOSE) exec -T backend python -m app.scripts.seed
	$(E2E_COMPOSE) exec -T backend python -m eval.demo_data
	@echo "app: http://localhost:3300   (demo-officer@example.test / demo-admin@example.test, password in eval/demo_data.py)"

e2e-down:
	$(E2E_COMPOSE) down -v

e2e:
	cd frontend && npx playwright test
