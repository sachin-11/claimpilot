# ClaimPilot

Health-insurance claims copilot: RAG over policy wordings (with clause citations), LangGraph
agents with human approval, and MCP servers. Built module by module; see [CLAUDE.md](CLAUDE.md)
for the roadmap. Currently: **Module 0 (setup and foundations)**.

## Run

```bash
cp .env.example .env        # then fill in OPENAI_API_KEY (and GROQ_API_KEY for fallback)
make up                     # postgres + redis + backend on http://localhost:8000
make migrate                # alembic upgrade head (enables pgvector)
curl localhost:8000/health
curl localhost:8000/health/ready
```

## Develop

| Command | What |
|---|---|
| `make test` | pytest (no real API calls) |
| `make lint` / `make fmt` | ruff check / format |
| `make revision m="msg"` | new Alembic autogenerate revision |
| `make shell` | bash inside the backend container |
| `make logs` | follow backend logs |

Tests and lint can also run on the host: `cd backend && uv venv && uv pip install -e ".[dev]"`
then `pytest` and `ruff check .`.
