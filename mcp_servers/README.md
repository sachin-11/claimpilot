# MCP servers (Module 7)

Three [MCP](https://modelcontextprotocol.io) servers built with the official Python SDK
(`mcp==1.30.0`, FastMCP). They reuse the backend services (`RetrievalService`, repositories), so
this folder holds only the tool layer. Each server runs over **stdio** (Claude Desktop, MCP
Inspector) or **Streamable HTTP** (the LangGraph agents in Module 8).

| server | port | purpose | writes |
|---|---|---|---|
| `policy_docs` | 8101 | policy wordings: clause search, grounded answers, structured rules | none |
| `claims_db` | 8102 | claims, extractions, customer history | `update_claim_status`, `save_decision_draft` |
| `notify` | 8103 | customer messages (delivery mocked, stored in `notifications`) | `send_status_update` |

## Tools

**policy_docs**: `list_policies()`, `search_policy(query, policy_code?, top_n=5)` (1-10, clauses with
`section_path` / pages / score, content truncated), `answer_policy_question(question, policy_code?)`,
`get_policy_rules(policy_code)`. Resources: `policy://{policy_code}/sections`,
`policy://{policy_code}/chunk/{chunk_id}`.

**claims_db** read: `get_claim`, `get_claim_extraction` (with `consistency_issues`,
`missing_documents`; patient name and source snippets omitted), `get_customer_history`
(previous claims, sum insured, remaining sum insured), `list_claims(status?, limit=20)`.
Write (only registered when `WRITE_ENABLED=true`):

- `update_claim_status(claim_number, new_status, reason, idempotency_key)`. Allowed:
  `submitted -> processing -> awaiting_approval | documents_requested`,
  `documents_requested -> processing`, any open status `-> failed`. **`approved`,
  `partially_approved`, `rejected` are refused**: only the human approval API (Module 8) sets them.
- `save_decision_draft(...)`: writes a draft `claim_decisions` row (claim must be `processing`; cited
  `chunk_id`s must exist; section / page come from our records). Changes no status.

**notify**: `send_status_update(claim_number, channel, template, idempotency_key, extra?)`. The
recipient is always the claim's customer on record (`extra` accepts only `missing_documents` and
`reason`; any other key, such as `to`, is ignored). `approved` / `partially_approved` / `rejected` are
sent only if the claim already has that status. Limit: `NOTIFY_MAX_PER_CLAIM_PER_HOUR` (5).

Write tools are idempotent per `idempotency_key` (same key and arguments return the first result with
`idempotent_replay: true`; same key, other arguments is an error) and write `audit_logs` rows with
`actor_type=agent`, `actor_id=<caller identity>`.

## Least privilege (what each Module 8 agent should get)

| agent | server.tool |
|---|---|
| Intake | `claims_db`: `get_claim`, `get_claim_extraction` (+ `update_claim_status` to move `submitted -> processing`, `documents_requested`) |
| Coverage | `policy_docs`: `list_policies`, `search_policy`, `get_policy_rules` (`answer_policy_question` optional) |
| Risk | `claims_db`: `get_claim`, `get_claim_extraction`, `get_customer_history`, `list_claims` |
| Decision | `claims_db`: `save_decision_draft`, `update_claim_status` (to `awaiting_approval`) |
| Notify | `notify`: `send_status_update`, **only after human approval** |

The client enforces it: `get_mcp_tools({"policy_docs": ["search_policy"]}, caller="coverage-agent")`
(`backend/app/mcp_client/client.py`). Each agent should use its own token so the audit log shows who
acted.

## Run

Docker (HTTP mode, ports above):

```bash
docker compose build backend            # once, the MCP SDK is in the backend image
docker compose up -d mcp-policy-docs mcp-claims-db mcp-notify
make mcp-smoke                          # lists all tools, calls search_policy / get_claim / get_claim_extraction
```

Tokens (`.env`, see `.env.example`): `MCP_<SERVER>_TOKENS="coverage-agent=tokA,risk-agent=tokB"`
(identity = token) or a single `MCP_<SERVER>_TOKEN`. HTTP mode refuses to start without one and
answers 401 to a missing or wrong bearer token (`/healthz` is open). `WRITE_ENABLED` per server:
`MCP_CLAIMS_DB_WRITE_ENABLED`, `MCP_NOTIFY_WRITE_ENABLED` (default true in compose, false elsewhere).

stdio (no token; identity from `MCP_CALLER_ID`, default `stdio-user`), via the running backend container:

```bash
docker compose exec -T backend python -m mcp_servers.claims_db.server --transport stdio
```

MCP Inspector (tool list; add `WRITE_ENABLED` through `env`, not `-e`, which the Inspector keeps):

```bash
npx @modelcontextprotocol/inspector --cli docker compose exec -T backend \
  env WRITE_ENABLED=true python -m mcp_servers.claims_db.server --transport stdio --method tools/list
```

Drop `--cli` for the web UI. Logs go to stderr in stdio mode (stdout is the protocol).

## Claude Desktop demo (read-only)

1. `docker compose up -d` (backend, postgres, redis) and run `make seed` / `make ingest` once.
2. Copy `claude_desktop_config.example.json` into Claude Desktop's config
   (`%APPDATA%\Claude\claude_desktop_config.json` on Windows), replace `<ABSOLUTE_PATH_TO_REPO>`,
   restart Claude Desktop. Both servers run with `WRITE_ENABLED=false`, so no write tool exists.
3. Ask: *"CLM-005 ka claim summary do aur check karo policy mein cataract ka waiting period kya hai"*.
   Claude calls `get_claim`, then `search_policy` / `answer_policy_question` and cites the clause.

Every tool call is logged as one JSON line (`tool`, `caller`, `latency_ms`, `ok`); arguments and
results are never logged.
