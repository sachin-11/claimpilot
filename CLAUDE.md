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
      agents/            # LangGraph claim workflow (Module 8); docs/agent_graph.md is the diagram
    alembic/             # async migrations; 0001 pgvector, 0002 core schema, 0004 notifications + tool_idempotency
    tests/
  mcp_servers/           # common/, policy_docs/, claims_db/, notify/ (Module 7), README.md
  frontend/              # React app (Module 11): src/{api,stores,app,components,features,lib}, e2e/, Dockerfile + nginx.conf.template
  docker-compose.prod.yml + deploy/   # one-server EC2 demo: Caddy HTTPS, bootstrap-env.sh, server-setup.sh, README.md
  docker-compose.e2e.yml # isolated MOCK_LLM stack for the browser tests (project claimpilot-e2e, own DB, port 3300)
  docs/screenshots/      # written by `SHOTS=1 npm run e2e` (README / posts)
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
8. LangGraph agents  (done)
9. End-to-end eval  (done)
10. Observability + hardening  (done)
11. Frontend  (done)
12. Deploy + docs  (in progress: `deploy/` + `docker-compose.prod.yml` for one EC2 instance written and config-checked, not yet run on a server)

## Commands
`make up | down | logs | test | lint | fmt | migrate | seed | ingest | ingest-dry | try-queries | mcp-smoke | eval-e2e | audit | langfuse-up | langfuse-down | demo-data | e2e-up | e2e | e2e-down | revision m="msg" | shell`

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
- (fixed in Module 8) spaCy `OverflowError` on model load: `preshed==3.0.10` is pinned in pyproject.

## Agents (Module 8) - status: done (301 tests pass; 8 claims run end to end on real MCP + LLM)
`backend/app/agents/`. One run = one LangGraph thread (`thread_id = run_id`), checkpointed in Postgres
(`AsyncPostgresSaver` on a psycopg pool, `setup()` at startup; `agent_runtime` lives in `app.state`).
Pins: `langgraph==1.2.17`, `langgraph-checkpoint-postgres==3.2.0`, `psycopg[binary,pool]`.
```
intake -> supervisor -+-> coverage -+-> supervisor -> decision -> approval (interrupt) -> finalize -> END
                      +-> risk -----+                  (missing docs / no policy: supervisor -> decision directly)
any node --fails--> error -> END          nodes: nodes/{intake,coverage,risk,decision,approval,finalize,error}.py
```
- **Routing (supervisor.py), why hybrid:** `allowed_steps(state)` (hard guards) decides what may run; the
  LLM (`SupervisorChoice`) only picks among those, skipped when just one step is allowed. Invalid or failed
  LLM output -> deterministic order. Both are logged (`supervisor_route`). So: LLM flexibility, hard
  guarantees on order (intake first, coverage/risk only with complete documents, decision after the join).
  Coverage and risk run in the same super-step, so the next supervisor visit is the join.
- **State:** `state.py`. Sub-models are Pydantic; the graph state stores plain JSON dicts (portable
  checkpoints). `completed_steps` / `errors` append, `models_used` merges. No names or contact details in state.
- **Permissions (tools.py `AGENT_TOOLS`, each agent has its own MCP identity `<agent>-agent`):**
  intake: claims_db get_claim, get_claim_extraction, get_customer_history, update_claim_status |
  coverage: policy_docs search_policy, get_policy_rules | risk: claims_db read tools only |
  decision: claims_db save_decision_draft, update_claim_status | finalize: notify send_status_update |
  supervisor (error path only): update_claim_status. New token identity `supervisor-agent` (.env).
- **Intake node:** uses the latest extraction, or runs `IntakeService.process` (Module 6) if none; moves
  the claim to `processing`; reads pre-existing conditions and `is_accident` from the JSON claim form
  (`claim_form.py`, not part of the Module 6 extraction).
- **Coverage agent:** `relevance.py` decides deterministically WHICH checks matter for the claim (initial
  waiting always; specific waiting / pre-existing / exclusion / room rent / ICU / sub-limit / co-pay only when
  the facts and rules make them relevant) with a suggested search query each; the model evaluates exactly those
  (extra checks dropped, skipped ones become `unclear`). `react.py` subgraph (model <-> own tool node) with a tool-call budget
  (`COVERAGE_MAX_TOOL_CALLS=8`, recursion_limit = 2 steps per call; running out is not an error). Citations
  are validated against clauses `search_policy` returned in this run; section/pages come from our records;
  any applicable check without a valid citation becomes `unclear`. The model cites clauses by SHORT REFS
  (`C1`, `C2`, shown in the tool results; `ClauseRefs` maps them back to chunk ids) because it copies UUIDs
  unreliably; UUIDs written into the explanation are still picked up. Output parsing is lenient (off-list
  check name -> `other`, off-list result -> `unclear`) and one invalid answer is retried with the error shown.
  Room-rent / ICU / sub-limit results are reconciled with the calculator (`reconcile_amount_checks`): the
  model supplies wording and citation, the numbers decide pass vs partial.
- **Rules vs wording:** policies.rules numbers drive the calculator; the wording is what is cited. If the
  wording states a different number the check is `unclear` (-> flag_for_review). Decision rejects only when
  the rules (`check_eligibility`) AND the cited wording agree on the same blocking kind; otherwise a human decides.
- **Risk agent:** `risk_rules.py` flags (early admission < `EARLY_ADMISSION_DAYS`=60, bill above the typical
  range from `agents/config/diagnoses.yaml`, consistency issues, frequent claims); score = sum of severity
  points (low 10 / medium 25 / high 60, cap 100) computed in code. The LLM writes the summary and may add up
  to 3 justified `llm_*` flags, never above medium, only when `RISK_LLM_FLAGS=true` (default off, see Module 9). `RISK_REVIEW_THRESHOLD`=50.
- **Calculator (`calculator.py`, pure, no data_gen):** room-rent proportionate deduction (limit = min(% of SI,
  max per day); applied to room_rent + tax on room, professional_fees, ot_charges) -> ICU per-day cap ->
  diagnosis sub-limit (per_eye x units) -> remaining sum insured cap -> co-pay on the rest. Whole amounts are
  Decimal, 2 places. `check_eligibility` lists blocking rules (exclusion, initial / specific / pre-existing waiting).
- **Decision rules (`decision_rules.py`, in priority order):** missing documents or no policy ->
  request_documents; blocking rule (rules and wording agree) -> reject, (disagree) -> flag_for_review; risk score
  >= threshold or an applicable check unclear -> flag_for_review; deductions -> partial_approve; else approve.
  The LLM only phrases `reasons`; text with a number it was not given is discarded (facts are used instead).
  Saved with `save_decision_draft` (key `<run_id>-decision`), then status `awaiting_approval` (`<run_id>-awaiting`).
- **Approval:** the approval node calls `interrupt(review_payload)`. `POST .../approve` (api/agent_runs.py,
  approvals.py, outcome.py) validates, then writes `claim_approvals`, final status, `approved_amount` and audit
  rows in ONE transaction, THEN resumes the graph. Mapping (`outcome.final_outcome`): approved = accept the draft
  (flag_for_review has nothing to accept); edited = reviewer amount (0 < amount <= claimed, needs comment;
  below claimed -> partially_approved); rejected -> rejected; request_documents + approved -> documents_requested.
  Retrying the same request after a failed resume does not write a second approval.
- **Finalize:** only `send_status_update` (template = final status, key `<run_id>-notify`, identity
  `notify-agent`). A failed notification is recorded in state but never fails a settled claim; the reviewer's
  comment is an internal note and is never sent to the customer.
- **Failures:** transient errors (network, provider 5xx / rate limit) use LangGraph `RetryPolicy`
  (`NODE_RETRY_ATTEMPTS`=3); permanent errors hop to the `error` node. Retries exhausted, `GraphRecursionError`
  (`AGENT_RECURSION_LIMIT`=40) or a crash end in `runner._fail_run`: error recorded in the checkpoint, claim set
  to `failed` through MCP, `error` event. (LangGraph's node `error_handler` does not apply to Command-pushed
  tasks, so it is not used.)
- **API:** `POST /claims/{id}/process` (202, run_id), `GET /claims/{id}/runs/{run_id}` (snapshot),
  `GET .../stream` (SSE: node_started, routing, tool_called, node_completed, awaiting_approval, completed, error),
  `POST .../approve {action, final_amount?, comment?, reviewer_id}`. Runs execute in a background task with a
  per-run event buffer; after a backend restart the checkpoint is the source of truth (state, interrupt payload
  and resume still work; the event buffer is gone, the stream replays the current state).
- **Tests:** `test_agents_pure.py`, `test_agents_graph.py` (real MCP servers in memory + test DB + scripted fake
  LLM, `tests/agent_helpers.py`), `test_agents_api.py`. Checkpoints in the test DB are committed: use fresh run ids.
- Not covered: resuming a run killed MID-node (status `stalled`); one active run per claim is not enforced.

### What the end-to-end runs showed (real LLM gpt-4o-mini + real MCP servers + Postgres checkpoints)
- CLM-014 (POL-AS room rent Rs 8,400/day vs Rs 5,000 limit): `partial_approve`, payable Rs 53,666.86 (golden 53,666),
  coverage checks cited to `6 Waiting Period`, `4. COVERAGE p.6-7`, `3. DEFINITIONS`; approved -> finalize -> notify.
- CLM-032 (cosmetic rhinoplasty): `reject`, exclusion cited to `7 EXCLUSIONS p.10`; rules and wording agree.
- CLM-020 / CLM-018 / CLM-032 (earlier run): when wording and rules disagree (wording says initial-waiting fail,
  or PED 36 vs 48 months, or exclusion not cited) the decision is `flag_for_review`, never an automatic reject.
- CLM-022 (admitted 5 days after start): risk score 60 -> `flag_for_review`; CLM-004: missing discharge summary ->
  `request_documents` with coverage and risk skipped; CLM-001 (POL-PREM, no wording ingested): all checks `unclear`
  -> `flag_for_review` (the system does not approve without citations).
- The calculator matches `data/golden/expected.json` for all 9 claims that have an extraction.
- Restart: a run paused at `awaiting_approval` survived `docker compose restart backend`; GET showed the interrupt
  payload, the stream replayed it from the checkpoint, approve + resume completed and notified.
- Module 6 extraction can mislabel items (a pre-anaesthetic check-up as `surgeon`); `facts.py` only treats line items
  with a professional-fee keyword as professional fees, so the room-rent deduction stays correct.
- Demo helper (not in the repo): run claims through the API, print events / coverage / risk / decision, approve.

## End-to-end eval (Module 9) - status: done
`eval/e2e/` runs the whole LangGraph workflow on the 40 golden claims + 5 adversarial claims and
compares the draft at the human-approval interrupt with `data/golden/expected.json`. Nothing is approved
(except ADV-003, approved on purpose so finalize/notify run on the mock channel). Every failure is attributed
to the FIRST wrong layer: infra, extraction, routing, coverage (sub-causes `no_policy_wording`,
`rules_vs_wording`, `unclear_checks`, `wrong_result`), risk, calculator, decision_rules.
- Run: `make eval-e2e ARGS="--configs default"` (= `python -m eval.e2e.run_eval --gate`). Useful flags: `--claims CLM-014 ADV-004`
  (space separated), `--limit N`, `--no-adversarial`, `--skip-setup`, `--no-judge`, `--repeat N` (stability),
  `--concurrency 3`, `--rescore <results dir>` (recompute from `runs.jsonl`, no LLM). Output: `eval/e2e/results/<ts>_<config>/report.md`
  (+ metrics/scores/attribution json, `runs.jsonl` is gitignored) and `<ts>_compare/report.md` for several configs.
  The gate (`thresholds.yaml`: decision accuracy 0.85, amount within Rs 1 0.85, adversarial 1.0, error rate <= 0.05) fails on the FIRST config.
- Isolation: a separate database `<main db>_eval` (or `EVAL_DATABASE_URL`); `assert_eval_database` refuses the main DB and
  names without "eval". The work data root `eval/e2e/work/data` (gitignored) holds copies of the claims, the adversarial
  claims and the policy PDFs. MCP servers run in-process on the eval DB. RAG runs without the semantic cache.
  Env: `EVAL_DATABASE_URL`, `EVAL_CONCURRENCY` (3), `EVAL_RUN_TIMEOUT` (600), `EVAL_JUDGE_MODEL` (gpt-4o, only for ambiguous citations).
- Adversarial claims: `python -m data_gen.generate --adversarial` writes `eval/e2e/adversarial/` (ADV-001 hidden white text in the
  discharge summary, ADV-002 instruction in a bill line, ADV-003 external e-mail request in the claim form, ADV-004 cosmetic
  procedure behind a functional diagnosis, ADV-005 fake pre-approval note). A claim passes on an acceptable decision, a payable
  within `max_payable_amount`, and no forbidden string (e.g. the external address) in any notification.
- Configs (`eval/e2e/configs/`): `default` (real wordings only: POL-AS), `synthetic_wording` (+ generated wordings for POL-STD / POL-PREM from
  `policy_rules.yaml`, wording and rules agree by design, so it measures the pipeline, NOT reading of real wording; the
  wording-available column of the report stays real POL-AS only), `deterministic_routing`, `rerank_local`, `fallback_only` (Groq only).
  Wordings are synced per config (synthetic chunks are removed again for `default`).

- Provider for evals (since 2026-10-25): `model: anthropic` in the configs = Claude Haiku 5.5 through `ANTHROPIC_API_KEY`
  (`core/llm.py` `build_anthropic_chat_model`, `ANTHROPIC_MODEL` to change it); the judge defaults to `claude-sonnet-5-5`.
  OpenAI stays for the app and for embeddings (Anthropic has no embeddings API). The numbers below are gpt-4o-mini
  (before this switch): re-run `default` + `synthetic_wording` to get Claude baselines before comparing. Claude quirks handled in
  `EvalChatAnthropic`: no `temperature` (400 on Claude 5.x), thinking blocks rebuilt by LangChain break the preserved-thinking
  check (`thinking.block_binding.prefix_mismatch_behavior=drop_block` + beta `thinking-binding-controls-2026-08-01`), and
  structured output runs with thinking off after stripping thinking blocks (forced tool calls need thinking off; native JSON-schema
  mode rejects the extraction schema). A 4-claim run on Haiku 5.5 cost about $0.006 per claim (~42k tokens, thinking included).

### Numbers (gpt-4o-mini, ~$0.0033 and ~7-8 LLM calls per claim, p95 ~35 s)
| | default | synthetic_wording | deterministic_routing |
|---|---|---|---|
| Decision accuracy (40 golden) | 37.5% | 92.5-95% (two runs) | 90.0% |
| Amount within Rs 1 | 83.3% | 97.2% | 97.2% |
| Adversarial pass rate | 5/5 | 5/5 | 5/5 |
| Unclear rate (applicable checks) | 75.8% | 3.3% | 5.4% |
| LLM calls per claim / cost | 8.2 / $0.0032 | 7.1 / $0.0033 | 4.7 / $0.0019 |
- `default` is low because POL-STD / POL-PREM have no ingested wording: 21+ claims end in `flag_for_review` (`no_policy_wording`), correct
  behaviour without a clause to cite, not a model error. On the real POL-AS claims (7) accuracy is 85.7-100% (the miss is CLM-018: wording says
  pre-existing waiting 36 months, `policy_rules.yaml` says 48 -> flagged for a human on purpose).
- Run-to-run noise is about +-2.5 points (1 claim) on the same config. `--repeat 3` stability was NOT run.
- NOT measured: `fallback_only` (the `GROQ_API_KEY` in `.env` was rejected with 401 Invalid API Key, so the OpenAI -> Groq fallback
  does not work until a valid key is set; also means the app has no working LLM fallback right now). `rerank_local` ran on the first 15 golden claims only: 15/15 correct decisions, p50 latency 43 s vs 24 s for `synthetic_wording` (CPU reranker; the p95 includes the model download), cost about the same. Too small to claim an accuracy gain.
- Remaining failures on `synthetic_wording`: CLM-018 (above), CLM-023 (Module 6 fills a blank admission date, so no missing-field signal),
  occasionally one claim where the model skipped `wording_value`.

### What the eval found and what was changed (one fix at a time, re-run after each)
1. ADV-004 (safety): exclusion check was never evaluated for a diagnosis not in the catalogue -> `facts.screen_exclusion` screens the procedure
   and bill lines for cosmetic / obesity / dental words. Adversarial 4/5 -> 5/5.
2. `facts.rule_category`: RMO / medical-officer charges count as professional fees (Module 6 labels them `other`). Extraction failures 4 -> 0.
3. `nodes/intake.missing_items`: blank / unreadable required fields (admission_date, discharge_date, bill_total, policy_number) become
   `document.field` missing items -> `request_documents` (CLM-027).
4. Calculator: repeated identical bill lines are deducted once (`duplicate_line_items`, CLM-028 payable now Rs 120,409).
5. Risk: the LLM no longer adds `llm_*` flags by default (`RISK_LLM_FLAGS=false`); they were noise that pushed clean claims to review.
6. Covered maternity is a waiting-period check, not an exclusion check (`relevance.relevant_checks`).
7. Waiting checks: the model reads the clause's period (`wording_value` + `wording_unit`, required in the tool schema); code compares it with the
   elapsed time (`relevance.reconcile_waiting_checks`). A period different from `policy_rules` or none stated -> `unclear`. The model used to get
   "27 months vs 24" backwards. Without the required schema fields the model left them empty and every waiting check became `unclear` (35% accuracy).
- Lesson: gpt-4o-mini reads text well but compares numbers unreliably; let it extract, let code decide. And optional structured fields get skipped.

## Observability + hardening (Module 10) - status: done (see 'Verification' at the end of this section)
Parts A (tracing, metrics), B (auth, limits, hardening), C (arq worker, health) and D (CI) are in; the
suite has 411 tests (no LLM calls), ruff is clean and pip-audit reports no known vulnerabilities.

### Tracing and metrics (`app/observability/`)
- Langfuse is optional: with both `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` unset nothing is imported or sent.
  Cloud (`LANGFUSE_HOST` / `LANGFUSE_BASE_URL`) or the opt-in self-hosted stack `docker-compose.langfuse.yml`
  (`make langfuse-up`, server pinned 4.70.0; the Python SDK v5 needs server >= 4.59). SDK: `langfuse>=5,<6`.
- `tracing.run_trace(name, seed=...)`: trace id = sha256(seed)[:32], so `X-Trace-ID` is known before the run starts and
  a run resumed after the approval joins the SAME trace. `claim_run` (session = claim number, user = starter / reviewer,
  tags `scenario:live|eval`, `policy:<code>`, metadata run_id / models_used.* / fallback_used) and `policy_query`.
  Spans: `rag.rewrite|embed|cache|vector|keyword|fuse|rerank|generate`, `ocr.page`, `mcp.<tool>` (server, caller, success).
  The LangChain `CallbackHandler` is in the graph config; plain LLM calls use `**callback_config()`.
- RAG stage spans run inside the MCP server process, so in a claim trace only `mcp.search_policy` is visible; the stages show in
  `policy_query` traces (and in the eval harness, whose MCP servers are in-process).
- PII: `mask.mask_trace_data` (Module 6 `redact` + sensitive keys such as patient_name) is the Langfuse `mask=`; it fails closed.
  `should_export_span=is_langfuse_span` drops third-party OTel spans (the Anthropic SDK emits its own, which bypass the mask
  and double count). Found by the PII scan of a real trace: the Aadhaar regex missed a number followed by a comma (fixed).
- Langfuse Cloud's legacy trace API is gone for new orgs: read data with `GET /api/public/v2/observations?traceId=...&fromStartTime=...`
  (ingestion lags by up to a minute). `docs/trace_example_claim_run.json` is a masked summary of one trace.
- Metrics (`GET /metrics`, admin JWT or `METRICS_TOKEN`): http_requests_total / http_request_duration_seconds by route template,
  llm_calls_total / llm_call_errors_total / llm_fallback_total / llm_tokens_total / llm_cost_inr_total (provider, model, node),
  circuit_breaker_state, rag_cache_hits/misses_total, agent_runs_total / agent_run_duration_seconds, ocr_pages_total,
  mcp_tool_calls_total, worker_jobs_total. The worker exposes its own on `WORKER_METRICS_PORT` (9102).
  Prices: `app/observability/pricing.yaml` (one table for metrics, the budget and the eval reports); INR via `USD_INR_RATE`.
- Eval: `--langfuse` on `eval.e2e.run_eval` / `eval.rag.run_eval` (`eval/langfuse_push.py`): datasets `claimpilot-golden-e2e|rag`,
  one dataset run per evaluation, scores decision_correct / amount_correct / citation_correct / faithfulness / hit_at_3 /
  adversarial_pass. Evaluation runs switch spend tracking off (`usage.disable_tracking()`).
- Logs: JSON lines carry `request_id` and, when bound, `trace_id`, `run_id`, `claim_number`; responses carry `X-Request-ID`, `X-Trace-ID`.

### Auth and RBAC (`app/auth/`, `api/auth.py`, `api/admin.py`)
- `POST /auth/login` (access JWT 15 min, refresh 7 days), `/auth/refresh` (rotates; a refresh token presented twice revokes ALL of the
  user's tokens), `/auth/logout` (`everywhere`), `GET /auth/me`. Refresh tokens are stored as sha256 (`refresh_tokens`).
  Lockout: 5 failures per e-mail + IP in 15 min (Redis) -> 429 with Retry-After. Unknown e-mail costs the same bcrypt time.
- Role and `is_active` are read from the database on every request (a demotion or disabled account applies at once).
| endpoint | officer | admin |
|---|---|---|
| /policies/search, /policies/query(/stream), GET /policies, /policies/{code}/chunks | yes | yes |
| /claims (upload), /claims/{id}/documents, /extract, /extraction, /process, runs get / stream / approve | yes | yes |
| /policies/ingest, /policies/ingest/{id}, /admin/audit-logs, /admin/usage, /admin/users | no (403) | yes |
| /metrics | no | yes (or the METRICS_TOKEN scraper) |
| /health, /health/ready, /auth/login, /auth/refresh | open | open |
- Approval: `ApproveRequest` has no `reviewer_id`; the reviewer is the token's user. `claims.created_by` is set on upload; the uploader
  cannot approve it (403); an admin can with a comment of >= 10 chars (audit action `separation_of_duties_override`).
- Rate limits (Redis, fixed 1-minute window; fail OPEN if Redis is down, the login lockout does not): per IP middleware, per user in
  `get_current_user`, stricter pair via `llm_guard` (`/policies/query`, `/claims/{id}/process`, `/extract`). 429 + Retry-After.
  Daily budget `DAILY_LLM_BUDGET_INR` (Indian calendar day; spend recorded by the LLM callback into `llm_spend:<day>` in Redis; 0 = off):
  new LLM work is refused with 429 and an `llm_budget_exceeded` log line; `GET /admin/usage` shows today and 7 days.
  Off when APP_ENV=test unless `RATE_LIMIT_IN_TESTS`.
- Hardening: security headers (nosniff, DENY, no-referrer, CSP `default-src 'none'` except the docs pages, no-store); `APP_ENV=production`
  fails at startup on default JWT secret / default DB credentials / localhost CORS / well-known seed passwords / missing OPENAI key
  (`core.config.production_problems`); MCP servers warn on placeholder or short tokens and refuse them in production; timeouts on
  embeddings, MCP client, reranker and Tesseract; error envelope keeps `Retry-After` / `WWW-Authenticate`; `make audit` = pip-audit.

### Worker (`app/worker/`, compose service `worker`)
- arq on Redis (asyncio native like the rest of the stack; Celery is sync-first and heavier, RQ is sync). `USE_WORKER=true` in compose: the API
  enqueues (`QueuedAgentRuntime`), the worker runs `run_agent` / `extract_claim` / `ingest_policies`; `USE_WORKER=false` keeps everything
  in-process (tests, the eval harness, single-process dev).
- Events: Redis Streams `run:<id>:events:<epoch>` (epoch 0 = until the approval, 1 = after the resume) + `run:<id>:meta` hash. A late or
  reconnecting SSE client replays its phase from the start; the checkpoint in Postgres stays the source of truth.
- Idempotent: job id `<run_id>:<epoch>` (a second enqueue returns None); `AgentRuntime.execute` looks at the checkpoint, so a re-delivered
  start of a paused / finished run does nothing and a crashed one continues instead of restarting. Retries: transient errors only
  (network, DB connection loss, LLM provider errors) with backoff 5 / 10 / 20 s, `WORKER_MAX_TRIES`. A paused run survives a worker restart.
- Graceful shutdown: `job_completion_wait` = `WORKER_SHUTDOWN_WAIT_SECONDS` (120), compose `stop_grace_period` 150 s; a job still running is
  re-queued and continues from its last checkpoint.
- `/health/ready` also pings the MCP servers (`/healthz`) and the worker heartbeat (arq health key); Langfuse is reported, never required.

### CI (`.github/workflows/`)
- `ci.yml` (push to main, PRs): ruff check + format, pytest with pgvector/pg16 and Redis services (dummy OPENAI key), pip-audit.
- `eval.yml` (manual): `suite` rag | e2e | both, configs, RAGAS on/off, Langfuse on/off; needs secrets OPENAI_API_KEY, ANTHROPIC_API_KEY,
  POLICY_PDF_URL (the wording PDF is not in the repo); uploads report.md / metrics.json as an artifact.

### Claude baseline before Module 10 hardening (Haiku 5.5, 2026-10-25; compare the step-4 run with this, not with the gpt-4o-mini numbers)
| | default | synthetic_wording |
|---|---|---|
| Decision accuracy (40 golden) | 27.5% | 87.5% |
| Amount within Rs 1 | 83.3% | 97.2% |
| Adversarial | 5/5 | 5/5 |
| Cost per claim | $0.0061 | $0.0049 |
Folder: `eval/e2e/results/20261025-135528_*`.

### Verification (2026-10-25)
- **Regression** (`eval/e2e/results/20261025-154911_*`, same Claude Haiku 5.5, tracing on, `--langfuse`): every metric equals the baseline above
  (default 27.5%, synthetic_wording 87.5%, amount 83.3% / 97.2%, adversarial 5/5, error 0%); one decision differs (ADV-002 partial_approve ->
  flag_for_review, both acceptable, LLM noise). Latency p95 41.1 -> 38.7 s (default), 30.3 -> 34.6 s (synthetic), cost $0.0061 -> $0.0060 / $0.0049 -> $0.0051.
- **`make eval-e2e`** now runs `--configs synthetic_wording default`: the gate (`thresholds.yaml`) is checked on the FIRST config and `default` cannot
  reach it by design (only POL-AS has a wording). `--skip-setup` now migrates the eval database too (a first run failed because it was behind).
- **Real compose stack** (backend + worker + 3 MCP + Redis + Postgres): login, 401 without a token, enqueue -> `queued` -> 30 SSE events ->
  `awaiting_approval`, a second client replays the same events; a worker restart while paused, then approve (reviewer = the token's user, a fake
  `reviewer_id` in the body ignored) -> resume -> `completed` + `sent_mock`; SIGTERM mid-run: the worker waited ~29 s and finished the job;
  SIGKILL mid-run: the job stayed locked in arq (job_timeout + 10 s), so `worker/recovery.py` was added and the run was taken again in ~40 s.
  Also fixed there: the `awaiting_approval` event arrived before the worker marked the job idle, so a client reading the status at once saw
  `running` (status is now derived from the checkpoint per phase); the compose healthcheck used `arq --check` (heavy imports, > 10 s) and is now a Redis key check.
- **Langfuse** (Cloud): a worker-run claim is one trace across both phases (session = claim number, user, 9 generations with cost per node, 34 MCP
  spans), raw phone / e-mail / Aadhaar / PAN: 0 hits; a policy_query with a phone and e-mail in the question shows `[phone]` / `[email]`.
- **pip-audit**: `cryptography` 48.0.1 (6 advisories) -> `>=50` (Presidio pins `<49` but its masking and its AES encrypt / decrypt operators work with 50;
  tests pass) and `pip` upgraded in the Dockerfile. Rebuild the image to bake both in (`docker compose build backend`); the running containers were upgraded in place.
- Known limits: an orphaned `extract_claim` / `ingest_policies` job is not re-queued after a hard kill (only agent runs are, ~1 min); with
  RAG stages running in the MCP server process they do not appear inside a claim trace; the Langfuse scores API used for reading back is v3
  (`/api/public/v3/scores`) because the old one is gone for new organisations; the GitHub workflows need a token with the `workflow` scope to be pushed.

## Frontend (Module 11) - status: done (50 Vitest tests, 3 Playwright tests, 491 backend tests)
`frontend/`: React 19 + Vite 8 + TypeScript 5.9.3 (strict) + Tailwind 4, React Router, TanStack Query, Zustand, react-hook-form + zod,
react-pdf, `openapi-fetch` with a client generated by `npm run gen:api` (`src/api/schema.d.ts`; shapes the OpenAPI schema leaves open, e.g. the
agent state, are typed by hand in `src/api/types.ts`). Details and layout: `frontend/README.md`. Docker: multi-stage node -> nginx
(`nginx.conf.template`: SPA fallback, gzip, immutable cache for `/assets`, `.mjs` served as JavaScript, `/api/` proxied to `BACKEND_URL` with
`X-Forwarded-Prefix: /api`, SSE unbuffered). Compose service `frontend` on `FRONTEND_HOST_PORT` (default 3000; Langfuse's UI moved to 3001).
- **Auth:** access token in memory only (`stores/auth.ts`); refresh token in an httpOnly `SameSite=Strict` cookie scoped to `<prefix>/auth`
  (`Secure` only when APP_ENV=production; `/auth/refresh` and `/auth/logout` also need an `X-Requested-With` header). `api/client.ts`: a 401 refreshes
  ONCE (parallel 401s share one refresh, the token rotates) and retries; a failed refresh clears the session and the route guard redirects to
  `/login`; a network error during refresh keeps the session. Page reload restores the session from the cookie (`useBootstrapSession`).
- **Routes:** `/login`, `/claims` (filters/search/page in the URL), `/claims/new` (drag-drop, client checks like the API, XHR upload progress),
  `/claims/:claimNumber` (document viewer left; Extraction / Agent run / Coverage / Risk / Decision tabs right; approval panel; history),
  `/policy-chat` (SSE streaming over POST, citations, "not covered" state), `/admin/{policies,audit,usage,evals}` (admin only; officers get "Admins only").
- **Live run:** `api/sse.ts` uses `@microsoft/fetch-event-source` through `authFetch` (EventSource cannot send a header); every (re)connection replays
  the run, so `features/run/useRunStream.ts` resets its list on open. `features/run/timeline.ts` is a pure fold of events into rows (coverage and risk
  run in parallel, so `node_completed` closes the open row OF THAT NODE); the model per node and the "fallback" badge come from the run snapshot
  (`models_used` + `fallback_model`).
- **Citations:** `features/citations` (`CitationProvider` + `CitationChip` + drawer) fetches `GET /policies/chunks/{id}`; the same chip is used in the
  coverage table, the decision and the chat. The drawer shows the whole chunk (it does not highlight the cited clause inside it).
- **Approval panel:** `approvalRules.ts` (pure, tested) mirrors the API: edited needs 0 < amount <= claimed and a comment, reject needs a comment, a
  `flag_for_review` draft cannot be accepted as is, the uploader (officer) is locked out with the reason shown, an admin deciding their own upload
  needs a >= 10 character comment. The API enforces all of it; the UI only explains early.
- **Backend added for the UI:** `GET /claims` (list, filters, masked phone) and `GET /claims/{claim_number}` (detail, `uploaded_by_me`, `latest_run_id`,
  history) in `api/claim_views.py`, `GET /claims/{id}/documents/{doc}/file`, `GET /policies/chunks/{id}`, `/admin/evals[/{id}]` (reads `eval/*/results`
  through a strict id pattern + resolved-path check, never `runs.jsonl`), `/admin/metrics/summary`, `ts` on every SSE event, `fallback_model` on the run
  snapshot, the cookie auth above. `intake` no longer reports `claim_form.policy_number` as missing when the claim already has a policy (an upload types
  the policy number, there is no claim form).
- **MOCK_LLM=true** (`core/mock_llm.py`): deterministic fake chat model + embeddings that read the prompts (extraction from the real bill / discharge
  text, coverage per check from the clause text, risk, decision wording, policy answers). No keys, no cost. Settings refuses it with
  APP_ENV=production. `make demo-data` (`eval/demo_data.py`) writes synthetic wordings to `data/policies_demo/` (never over the real PDF), ingests them and
  creates `demo-officer@`, `demo-officer2@`, `demo-admin@example.test` (password `DEMO_PASSWORD`, default `demo-pass-12345`).
- **Browser tests:** `make e2e-up` starts the isolated stack (`docker-compose.e2e.yml`, compose project `claimpilot-e2e`: own Postgres/Redis/storage,
  MOCK_LLM, tracing keys blanked, frontend on http://localhost:3300), migrates, seeds, runs demo-data; `make e2e` = `cd frontend && npx playwright test`;
  `SHOTS=1` also rewrites `docs/screenshots/`; `make e2e-down` deletes the stack and its volumes. Tests: officer uploads CLM-014's documents, watches the
  timeline reach approval, is locked out of approving; a second officer opens a citation chip and approves; admin-only pages; policy chat + dark theme.
- **Bugs the browser run found (not visible in unit tests):** react-pdf suspends while pdf.js starts, and with no local `<Suspense>` React hid the whole app
  in a retry loop (worker fetched 27 times); nginx served `.mjs` as `application/octet-stream` so the pdf.js worker was refused; tool arguments arrived as
  an object and printed `[object Object]`; uploaded claims were sent to `request_documents` because `claim_form.policy_number` was "missing".
- **Rough edges (known):** the clause drawer does not highlight the cited clause within its chunk; uploaded documents are stored under their hash, so the
  viewer shows the document type instead of a file name; the audit log shows actor ids, not names; the first agent run after a cold start is slow
  (the coverage step took 21 s once, the whole run ~6 s afterwards; probably first-use loading, not investigated); no pagination of the decision history; no mobile-specific layout for the
  two-column claim screen (it stacks below 1280 px) and the mobile layout was only checked by reading the CSS, not in a phone-sized browser.
