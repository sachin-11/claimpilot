# ClaimPilot: interview notes

Personal prep sheet. Everything here is true of the code in this repo as of Module 8. Numbers marked
"measured" came from real runs; where I did not measure something I say so.

## 1. 30-second pitch

ClaimPilot is a health-insurance claims copilot. It reads a claim's documents (bill, discharge summary, claim
form), checks coverage against the real policy wording with clause citations (RAG), runs a LangGraph
multi-agent workflow (intake, coverage, risk, decision, with a supervisor), pauses for a human to approve, and
then notifies the customer. Data access goes through MCP servers, so each agent only gets the tools it needs.

The one-line design idea: **the LLM reads, searches and explains; code decides and calculates; a human signs off.**

## 2. Architecture in one picture

```
documents -> Module 6 intake (OCR, PII masking, LLM extraction, consistency checks)
                |
   intake -> supervisor -+-> coverage agent (ReAct, policy_docs MCP, citations) -+-> supervisor -> decision -> [human approval] -> finalize
                         +-> risk agent (rules + LLM summary, claims_db MCP) ----+        (missing docs: skip to decision)
   any failure -> error node (claim marked failed)
```
Stack: FastAPI + SSE, SQLAlchemy 2 async + Alembic, PostgreSQL 16 with pgvector (relational data and vectors in
one DB), Redis (semantic cache), LangGraph with Postgres checkpoints, MCP (FastMCP) servers, OpenAI primary with
Groq fallback, Docker Compose.

## 3. The decisions I should be able to defend

**One database for vectors and relational data (pgvector).** Fewer moving parts, transactional consistency
between a claim and its citations, HNSW cosine index plus a GIN full-text index in the same table.

**Hybrid retrieval, not just vectors.** Vector search and keyword search (Postgres full-text) are fused with
Reciprocal Rank Fusion (k=60), then optionally reranked by a cross-encoder. Measured finding: for "initial
waiting period" the keyword ranker put the right clause at #11 (the definitions section repeats "waiting
period" everywhere), RRF alone dropped it, and the answer was wrong. A local reranker (bge-reranker-base) fixed
it, at about 6 s per query on CPU, so the default stays "none" and the reranker is opt-in.

**Citations are validated in code, not trusted.** The model can only cite clauses that were actually retrieved
in this run. Section path and page numbers come from our database, not from the model. A conclusion without a
valid citation is downgraded to `unclear`.

**Short refs instead of UUIDs.** Models copy a 36-character UUID unreliably and a label like `C2` reliably. I hit
this twice (RAG answerer in Module 4, coverage agent in Module 8) and used the same fix: show the model `C1,
C2...`, map back to chunk ids in code.

**Hybrid supervisor.** Hard guards compute which steps are *allowed* (intake first; coverage and risk only when
documents are complete; decision only after both finished). The LLM may only choose among allowed steps. If its
answer is invalid or the call fails, we fall back to the deterministic order, and both are logged. If only one
step is allowed the LLM is not even called. Why: LLM flexibility with hard guarantees on ordering.

**Money never comes from an LLM.** `calculator.py` is a pure function: room-rent proportionate deduction, ICU
cap, sub-limit, remaining sum insured, co-pay. It matches the golden dataset on all 9 claims that had an
extraction (e.g. CLM-014: 53,666.86 vs golden 53,666). The LLM only phrases the reasons, and its text is thrown
away if it contains a number it was not given (then deterministic facts are used).

**The decision type is rule-based** (`decision_rules.py`): missing documents -> request_documents; a blocking
rule (waiting period, exclusion) rejects only if the policy rules AND the cited wording agree, otherwise
flag_for_review; high risk score or any unclear check -> flag_for_review; deductions -> partial_approve; else
approve. Conservative on purpose: when evidence is missing or contradictory a human decides.

**Rules vs wording can disagree.** The structured rules (`policies.rules`, from a YAML) and the PDF wording are
different sources (CLM-018: pre-existing waiting 48 months in rules, 36 in the PDF). The system does not pick
one silently; it flags the claim for review. That is a real-world data-quality problem I surfaced, not hid.

**Least privilege through MCP.** Three servers (policy_docs read-only, claims_db, notify). Each agent connects
with its own bearer-token identity and only receives its allowed tools (coverage cannot write; risk cannot
write; only decision saves drafts; only finalize notifies). The identity lands in `audit_logs` as actor_type
`agent`. Write tools do not even exist when `WRITE_ENABLED=false`.

**Agents cannot approve or reject.** The claim status state machine in the claims_db server refuses
approved / partially_approved / rejected for agents. Only the human approval API, in one DB transaction (approval
row + final status + approved amount + audit rows), can do that, and only then the graph is resumed.

**Idempotency everywhere that writes.** Every write tool takes an idempotency key (stored with a hash of the
request; same key + same request returns the first result, same key + different request is an error; an
advisory lock serialises concurrent calls). Run-scoped keys (`<run_id>-decision`, `-awaiting`, `-notify`) mean a
retried node never duplicates a decision or an email.

**Human-in-the-loop with real persistence.** LangGraph `interrupt()` pauses the graph; state is checkpointed in
Postgres (`AsyncPostgresSaver`, thread_id = run_id). Measured: I restarted the backend container while a run was
awaiting approval, then approved it and it finished and notified. Node code before `interrupt()` must have no
side effects because the node re-runs from the top on resume.

**Privacy.** PII masking (Presidio plus Indian-format recognisers) before any text reaches an LLM; logs go
through a redaction filter; MCP tools return summaries without contact details; the reviewer's internal comment
is never sent to the customer; the notify recipient comes from the customer record, never from tool arguments
(prompt-injection safe).

## 4. Bugs and surprises (these make good interview stories)

1. **Agents produced `unclear` for everything.** The model wrote correct explanations but did not copy the
   chunk UUID into the citation field. Fix: short refs (C1, C2), harvest ids written into prose, and an
   automatic retry that shows the model its validation error.
2. **LLM judged room rent "fail" while the calculator found no deduction.** Fix: for amount-type checks the
   calculator decides pass/partial; the model supplies wording and citation only.
3. **LLM said "initial waiting: fail" for a claim 161 days after policy start (30-day rule).** The safety net
   caught it: rules said no blocker, wording said blocker, so the decision became flag_for_review instead of an
   automatic rejection. Lesson: cross-check an LLM judgment against a deterministic computation.
4. **A bill line "Pre-anaesthetic Check-up" was extracted as a surgeon fee.** That inflated the proportionate
   deduction by about Rs 538. The calculator was right, the upstream label was wrong; fixed with a keyword guard
   for real professional fees. Found by comparing with the golden dataset.
5. **Tool events from inside a subgraph never reached the SSE stream.** A subgraph has its own stream; fixed by
   forwarding to the outer writer.
6. **LangGraph `error_handler` does not apply to tasks started with `Command(goto=...)`.** I verified it with a
   small experiment instead of assuming, removed it, and handled "retries exhausted" in the runner.
7. **A fresh run could 404 right after `POST /process`** because no checkpoint existed yet and the first
   checkpoint had no `claim_id`. Fixed by returning a `running` snapshot from the in-memory channel.
8. **Checkpoints are committed, tests are rolled back.** Re-running a test with a fixed thread id resumed an old
   finished thread. Fix: unique run ids per test.
9. **An environment bug from before my change:** spaCy failed to load its model (`OverflowError`) because
   `preshed 3.0.14` is incompatible; pinning `preshed==3.0.10` fixed it and 25 failing tests.
10. **Test infrastructure:** parallel graph nodes shared one asyncpg connection in tests, which the driver
    forbids; a lock makes the shared session behave like short sessions. A hung async test now fails after 90 s
    (pytest-timeout) instead of blocking the suite and holding database locks.

## 5. What I measured vs what I did not

Measured: calculator vs golden (9/9); full test suite (301 tests, no real LLM or network calls in tests);
end-to-end runs on real MCP servers + gpt-4o-mini for 8 claims; restart-survives-checkpoint test; reranker
latency (about 6 s/query CPU).
Not measured / not done: production-scale load; RAG answer quality beyond the Module 5 eval harness (results
are in `eval/rag/results/`); policies other than Arogya Sanjeevani have no ingested wording, so claims on
POL-STD and POL-PREM correctly end in flag_for_review; no auth system, observability or UI yet (Modules 10-11).

## 6. Honest limitations (say them before they ask)

- Only one policy PDF is ingested, so most automatic outcomes on other policies are "needs review".
- LLM output is probabilistic: I contain it (validation, retries, deterministic cross-checks, human approval)
  rather than claim it is always right.
- A run killed in the middle of a node is reported as `stalled` and is not auto-resumed.
- One active run per claim is not enforced.
- Delivery of notifications is mocked (rows in `notifications`, status `sent_mock`).
- The graph diagram is exported as Mermaid only (`docs/agent_graph.md`).

## 7. Likely questions and short answers

**Why not let the LLM decide the claim?** Money and compliance. A wrong number or an unexplained rejection is
costly, so the LLM never produces amounts or the decision type. It reads, searches, cites and explains.

**How do you stop hallucinated citations?** Retrieval returns real clauses; the model cites short refs; code
maps refs to chunk ids and drops anything not retrieved in this run; section and pages come from the database;
no valid citation means `unclear`, and `unclear` means a human reviews.

**Why MCP instead of calling the repositories directly?** One audited, permissioned tool boundary. The same
servers work for Claude Desktop (stdio) and for the agents (HTTP with bearer tokens). Least privilege and
per-agent identity are enforced at the server, not by trusting the agent's prompt.

**How does human-in-the-loop survive a restart?** LangGraph checkpoints every step to Postgres. The paused run
is just a row; any process can load it and resume with `Command(resume=...)`. I tested it by restarting the
container.

**What if the LLM provider is down?** `get_chat_model()` has OpenAI primary and Groq fallback with a circuit
breaker. Transient errors are retried by LangGraph retry policies; persistent failure ends the run on the error
path and the claim is marked `failed`. Embedding failure degrades retrieval to keyword-only.

**How would you scale it?** Run the MCP servers and API as separate replicas (they are stateless; state is in
Postgres/Redis), move the per-run event buffer to Redis pub/sub, add a queue for starting runs, and add
Langfuse for tracing (planned Module 10).

**What would you do next?** Ingest the other policy PDFs (then POL-STD and POL-PREM claims can be confirmed
with citations), run the Module 9 end-to-end eval against the golden set, add auth and observability, build
the reviewer UI.

**How did you test without calling an LLM?** Scripted fake chat models, real MCP servers connected in memory to
a rolled-back test database, pure-function tests for the calculator, rules and routing, and an API test client
in the same event loop.

## 8. Numbers worth remembering

53 policy chunks (about 19k tokens) for Arogya Sanjeevani; RRF k=60; reranker about 6 s/query on CPU; 40
claims in the golden set, 9 with extractions compared; 301 tests; 3 MCP servers on ports 8101-8103; global
recursion limit 40; coverage tool-call budget 8; risk review threshold 50 (single high flag = 60).

## 9. Where to look in the repo

- Agents: `backend/app/agents/` (graph.py, supervisor.py, nodes/, calculator.py, decision_rules.py)
- MCP servers: `mcp_servers/` (README has the tools and the permission matrix)
- Approval API: `backend/app/api/agent_runs.py`, `backend/app/agents/approvals.py`
- Graph diagram: `docs/agent_graph.md`
- Design record and findings: `CLAUDE.md`
