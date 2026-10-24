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
      ingestion/         # PDF -> chunks -> embeddings (Module 3)
      rag/               # hybrid retrieval + grounded answers (Module 4)
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
3. Policy ingestion  (done; only POL-AS has a PDF so far)
4. RAG retrieval  (done; manually verified on POL-AS only)
5. RAG eval
6. OCR + intake extraction
7. MCP servers
8. LangGraph agents
9. End-to-end eval
10. Observability + hardening
11. Frontend
12. Deploy + docs

## Commands
`make up | down | logs | test | lint | fmt | migrate | seed | ingest | ingest-dry | try-queries | revision m="msg" | shell`

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

## Policy ingestion (Module 3) - status: done for POL-AS (53 chunks embedded, 19k tokens)
`app/ingestion/`: parser (PyMuPDF) -> cleaner -> chunker (tiktoken cl100k) -> embedder -> pipeline.
CLI `python -m app.ingestion.run --policy POL-AS | --all [--force] [--dry-run]`; API under `/policies`.
- Parse is two phases with the cleaner in between (header/footer removal must precede heading
  detection). Headings: numbered, ALL CAPS, font/bold, plus line-level clause headings ("6.1. ...")
  because real wordings print them in the body font. Lone markers ("(i)", "7.4") are glued to the
  next line/block. Table cells are read with `get_text(clip=cell)` (Table.extract() drops spaces).
- section_path keeps clause numbers: `6 Waiting Period > 6.3. Specific Waiting Period`. Chunk metadata:
  policy_code, insurer, block_types, has_table, headings, clauses (e.g. ["7.5","7.6"]), source_pdf,
  embed_text (the text actually embedded: "Policy: <name> | Section: <path>

<content>").
- Defaults (env overridable): CHUNK_MAX_TOKENS=800, CHUNK_OVERLAP_TOKENS=120, CHUNK_MIN_TOKENS=80,
  EMBED_BATCH_SIZE=64, EXCLUDE_SECTION_PATTERNS=[] (.env sets ["get in touch with us"]).
- Unchanged-file check uses only the PDF hash: after changing chunk params use `--force`.
- Per-policy work runs in a SAVEPOINT; failure rolls back chunks, the `ingestion_runs` row is kept
  as failed. `pipeline.after_commit(result)` (call it after COMMIT) fires `on_policy_changed`.
- Quota errors (`insufficient_quota`) are not retried. Re-running an unchanged file is skipped;
  `--force` re-parses but embeds 0 chunks (embeds only chunks with embedding IS NULL).
- Only POL-AS has a PDF (`data/policies/arogya_sanjeevani.pdf`, untracked). Known: policy_rules.yaml
  differs from that wording (PED 48 vs 36 months, joint replacement 24 vs 48, cataract cap, dental).

## Retrieval (Module 4) - status: done, manually verified on POL-AS
`app/rag/`; entry point `RetrievalService` (`service.py`), built by `build_rag_service(...)`.
- `search(query, policy_codes=None, top_n=None) -> list[RetrievedChunk]` (no LLM; for MCP/agents)
- `answer(query, policy_codes=None, chat_history=None, use_cache=True) -> PolicyAnswerResult`
- `answer_stream(...)` -> events `citations` -> `token`* -> `done` | `error`
- API: `POST /policies/search`, `/policies/query`, `/policies/query/stream` (SSE).
- Pipeline: rewrite (only with history) -> embed -> cache lookup -> vector (pgvector cosine, HNSW,
  iterative scan) || keyword (websearch_to_tsquery, content words OR-ed, ts_rank_cd) -> RRF (k=60)
  -> rerank -> grounded answer -> validate citations -> cache store. One `rag_request` log line.
- Resilience: embedding failure -> keyword-only (warning); reranker failure -> fused order (warning);
  cache errors never fail a request; degraded answers are not cached.
- Answer: structured output (`function_calling`), clauses in `<clause id n policy section pages>`
  tags (closing tags escaped), citations must be among the sent chunks (section/pages always come
  from our records). Streaming rebuilds citations from `[n]` markers (supporting_text = clause start).
- Cache: Redis, per policy filter, cosine >= 0.95, TTL 24h; invalidated per policy after ingestion.
- Config (env): VECTOR_TOP_K=20 KEYWORD_TOP_K=20 FUSED_TOP_K=20 RRF_K=60 RERANKER=none|local|cohere
  RERANK_TOP_N=5 MULTI_QUERY=false CACHE_SIM_THRESHOLD=0.95 CACHE_TTL_SECONDS=86400 (see rag/config.py).
- Local reranker is optional: `INSTALL_LOCAL_RERANKER=true docker compose build backend` (torch, ~1 GB
  model download on first use). Cohere needs COHERE_API_KEY.
- Manual check: `make try-queries ARGS="--reranker none"` or `ARGS="--compare none,local"`.

### Retrieval findings on POL-AS (RERANKER=none vs local, gpt-4o-mini answers)
- RRF alone dropped the right clause for "initial waiting period": vector ranked 6.2 (30 days) #1 but
  keyword ranked it #11 (definitions full of "waiting period"), so it fell to #6 and the answer wrongly
  said 24/36 months. `RERANKER=local` (bge-reranker-base) fixed Q5, fixed Q6 ("treatment abroad" is
  excluded by clause 7.19; `none` wrongly said "not covered by the text"), and removed day-care
  annexure noise from the top 5.
- Cost of local: ~5.8-6.7 s per query on CPU for 20 candidates (avg rerank 7.8 s incl. warm-up) vs
  embed ~0.6 s, vector/keyword ~7 ms each, generate ~2.2 s. To speed up: lower FUSED_TOP_K (e.g. 10)
  and/or RERANK_MAX_CHARS. Default stays RERANKER=none (no torch in the default image).
- The LLM cites clauses by short number (`clause_n`), not UUID: gpt-4o-mini wrote "9.3" as a chunk id
  and that used to fail the whole request. Unresolvable citations are dropped and logged.
