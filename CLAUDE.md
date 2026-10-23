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
      models/            # SQLAlchemy models (Module 2)
    alembic/             # async migrations; 0001 enables pgvector
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
0. Setup  (done here)
1. Data generation
2. DB schema
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
`make up | down | logs | test | lint | fmt | migrate | revision m="msg" | shell`
