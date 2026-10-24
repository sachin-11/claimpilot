# ClaimPilot

## Goal
ClaimPilot is a health-insurance claims copilot (portfolio project). It reads claim documents
(bill, discharge summary), checks coverage against real policy wordings using RAG with clause
citations, runs LangGraph agents (Intake, Coverage, Risk, Decision + a supervisor) with
human-in-the-loop approval, and exposes data access through MCP servers.

We build it **module by module** (see roadmap). Never build features from a later module early.

## Stack
- Backend: Python 3.11+, FastAPI, SSE streaming, SQLAlchemy 2.0 (async) + Alembic
- Agents: LangGraph (PostgresSaver checkpointer later)
- LLM: OpenAI primary, Groq fallback (LangChain), see `backend/app/core/llm.py`
- DB: PostgreSQL 16 + pgvector (one DB for relational data and vectors)
- Cache: Redis
- MCP: Python MCP SDK (FastMCP), later
- Frontend: React + Vite + TypeScript + Tailwind, later
- Eval: RAGAS. Observability: Langfuse (both later)
- Deploy: Docker Compose, later AWS

## Folder map
```
claimpilot/
  CLAUDE.md  README.md  docker-compose.yml  .env.example  Makefile
  backend/
    pyproject.toml  Dockerfile  alembic.ini
    app/
      main.py            # app factory, lifespan, middleware, routers
      api/               # routers (health.py now; more per module)
      core/              # config, logging, db, redis, llm, errors
      models/            # SQLAlchemy models + enums (Module 2)
      schemas/           # Pydantic Create/Read schemas
      repositories/      # thin async repositories (flush only, caller commits)
      scripts/seed.py    # python -m app.scripts.seed
    alembic/             # async migrations; 0001 pgvector, 0002 core schema
    tests/
  mcp_servers/           # MCP servers (Module 7)
  frontend/              # React app (Module 11)
  data/{policies,claims,golden}/
  eval/                  # RAGAS + end-to-end eval (Modules 5, 9)
```

## Conventions
- Type hints everywhere. Pydantic models for all I/O (requests, responses, settings, LLM
  structured output).
- DB access is async only (SQLAlchemy 2.0 `AsyncSession`, asyncpg). No sync engines in app code.
- No secrets in code. Everything comes from `.env` through `app/core/config.py`.
- Write tests for pure logic. Tests never call real LLM or network APIs.
- Get LLMs through `get_chat_model()` / `get_embeddings()`, never instantiate providers directly.
- Log through the `logging` module (JSON, carries `request_id`). No `print`.
- Lint/format with ruff (`make lint`, `make fmt`). Schema changes go through Alembic.

## Module roadmap
0. Setup  (done)
1. Data generation  (done)
2. DB schema  (done)
3. Policy ingestion
4. RAG retrieval
5. RAG eval
6. OCR + intake extraction
7. MCP servers
8. LangGraph agents
9. End-to-end eval
10. Observability + hardening
11. Frontend
12. Deploy + docs

## Commands
`make up | down | logs | test | lint | fmt | migrate | seed | revision m="msg" | shell`

## Database schema (Module 2)
All tables: UUID pk (`gen_random_uuid()`), `created_at`/`updated_at` (audit_logs has no
`updated_at`). Constraint naming convention lives on `Base` in `core/db.py`.
```
policies 1-* policy_chunks     (embedding vector(1536) + HNSW cosine; search_tsv GENERATED + GIN;
                                unique (policy_id, content_hash); python attr `meta` = column "metadata")
policies 1-* customer_policies *-1 customers
customer_policies 1-* claims   (claims.customer_policy_id is NULL if the claim has no policy number)
claims 1-* claim_documents | claim_extractions (version) | claim_decisions
claim_decisions 1-* claim_approvals *-1 users
audit_logs                     (no FKs: entity_type + entity_id; append-only via DB trigger)
```
- Native PG enums: claim_status, claim_source, doc_type, decision_type, approval_action,
  user_role, actor_type. Python enums in `models/enums.py` (decision values = Module 1's).
- Repositories flush but never commit. Every claim status change goes through
  `ClaimRepository.update_status`, which writes an `audit_logs` row in the same transaction.
- Seed (`make seed`) is idempotent, loads policies from `data_gen/policy_rules.yaml` and claims
  from `data/claims/*`, never `data/golden`. `claims.icd10_code` stays NULL (only in golden;
  Module 6 fills it). `customer_policies.end_date` is NULL (open-ended). Document paths are
  relative to `data_dir`. Seed users come from `SEED_*` env vars.
- LangGraph checkpointer tables are NOT here; `PostgresSaver.setup()` creates them in Module 8.
- Tests: `make test` (needs postgres). They use `<db>_test` (or `TEST_DATABASE_URL`), migrate once
  per session, and roll back per test. `data_gen` is not importable in the backend container;
  only `data_gen/policy_rules.yaml` is mounted (read-only at `/data_gen`).
