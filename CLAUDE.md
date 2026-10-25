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
      mcp_client/        # tokens.py, client.py (MultiServerMCPClient), smoke.py (Module 7)
    alembic/             # async migrations; 0001 pgvector, 0002 core schema, 0004 notifications + tool_idempotency
    tests/
  mcp_servers/           # common/, policy_docs/, claims_db/, notify/ (Module 7), README.md
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
7. MCP servers  (done)
8. LangGraph agents
9. End-to-end eval
10. Observability + hardening
11. Frontend
12. Deploy + docs

## Commands
`make up | down | logs | test | lint | fmt | migrate | seed | ingest | ingest-dry | try-queries | mcp-smoke | revision m="msg" | shell`

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

## MCP servers (Module 7) - status: done
`mcp_servers/` (top-level package; mounted at `/mcp_servers`, `PYTHONPATH=/app:/`, pytest `pythonpath=[".."]`).
Official SDK pinned `mcp==1.30.0` (+ `langchain-mcp-adapters==0.3.2`; mcp 2.x exists but the adapters need 1.x).
Details, tool lists, Claude Desktop demo: `mcp_servers/README.md`.
- Servers (FastMCP, each `python -m mcp_servers.<name>.server --transport stdio|http --host H --port P`):
  `policy_docs` 8101 (read-only), `claims_db` 8102, `notify` 8103. Compose services `mcp-policy-docs`,
  `mcp-claims-db`, `mcp-notify` reuse the `claimpilot-backend` image (rebuild it after dependency changes).
- Each server module exposes `create_server(deps: ServerDeps | None, *, host=...)`; tests inject a
  rolled-back DB session and a fake RAG service (`tests/mcp_helpers.py`, `tests/test_mcp_servers.py`).
- Tools: policy_docs `list_policies search_policy answer_policy_question get_policy_rules` (+ resources
  `policy://{code}/sections`, `policy://{code}/chunk/{id}`); claims_db read `get_claim get_claim_extraction
  get_customer_history list_claims`, write `update_claim_status save_decision_draft`; notify `send_status_update`.
- Write tools are not registered when `WRITE_ENABLED=false` (default; compose sets true for claims_db / notify).
- `claims_db/state_machine.py`: agents may do submitted->processing->awaiting_approval|documents_requested,
  documents_requested->processing, open->failed. approved / partially_approved / rejected are HUMAN-ONLY
  (Module 8 approval API sets them, never an MCP tool). `save_decision_draft` needs status `processing`,
  verifies cited chunk_ids, never changes status.
- Idempotency: `tool_idempotency` (tool, key) stores the first response; notify uses `notifications.idempotency_key`.
  Writes audit as `actor_type=agent`, `actor_id=<caller>`.
- notify: recipient only from the claim's customer record (`extra` allows just `missing_documents`, `reason`);
  approved/partially_approved/rejected templates need the claim to already have that status; mocked delivery
  (`sent_mock`); `NOTIFY_MAX_PER_CLAIM_PER_HOUR=5`. Templates: `mcp_servers/notify/templates/*.j2`.
- HTTP auth: bearer token, `MCP_<SERVER>_TOKENS="identity=token,..."` or `MCP_<SERVER>_TOKEN`; identity goes to
  the audit log; server refuses to start without a token; `/healthz` is open. stdio: identity `MCP_CALLER_ID`;
  logs go to stderr there. One `tool_call` JSON log line per call (no arguments / results).
- Client: `app.mcp_client.client.get_mcp_tools({"policy_docs": ["search_policy"]}, caller="coverage-agent")`;
  `make mcp-smoke`.

### Permissions matrix (what each Module 8 agent gets)
| agent | tools |
|---|---|
| Intake | claims_db: get_claim, get_claim_extraction, update_claim_status |
| Coverage | policy_docs: list_policies, search_policy, get_policy_rules (answer_policy_question optional) |
| Risk | claims_db: get_claim, get_claim_extraction, get_customer_history, list_claims |
| Decision | claims_db: save_decision_draft, update_claim_status (-> awaiting_approval) |
| Notify | notify: send_status_update, only after human approval |

### Known issues
- 25 intake/PII tests fail (`tests/test_intake_*`): `spacy.load("en_core_web_sm")` raises OverflowError in the
  current image (also in a pristine container). Not related to Module 7; 193 other tests pass.
