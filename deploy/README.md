# Deploying ClaimPilot on one EC2 instance (demo)

One server, Docker Compose, HTTPS through Caddy. Meant for a demo link you start, show and stop again.

```
internet --443--> Caddy --> nginx (web app, /api proxy) --> backend (FastAPI) --> Postgres + pgvector
                                                              |--> Redis <-- worker (agent runs)
                                                              '--> MCP servers (policy docs, claims DB, notify)
```

Only Caddy publishes ports (80, 443). Everything else lives on the compose network.

## What it costs (approximate, Mumbai `ap-south-1`; check the AWS Pricing Calculator)

| | per hour | note |
|---|---|---|
| `t3.large` (2 vCPU, 8 GB) | ~$0.09 | recommended: the reranker model runs in two processes |
| `t3.medium` (4 GB) | ~$0.045 | tight: the 4 GB swap file from `server-setup.sh` keeps it alive but slow |
| 30 GB gp3 disk | ~$3 / month | kept while the instance is stopped |
| Elastic IP | ~$3.6 / month | only while you keep it; a running instance's public IP costs the same |

A stopped instance costs only the disk (+ the Elastic IP if you keep it). **Terminated = $0.** OpenAI usage is
cents for a demo (about $0.003-0.006 per claim) and `DAILY_LLM_BUDGET_INR` caps it.

Set an AWS Budgets alert (for example $10) before you start.

## 0. Before you start

- A host name that points at the server. Free: a [DuckDNS](https://www.duckdns.org) sub-domain
  (`claimpilot-yourname.duckdns.org`). Or a domain you own. HTTPS needs a name, an IP address is not enough, and
  the login cookie is `Secure` in production.
- An OpenAI API key (chat model and embeddings).
- The policy wording PDF `data/policies/arogya_sanjeevani.pdf` on your laptop (it is not in the repo).

## 1. Create the server

Console or CLI, either way:

- AMI: Ubuntu Server 24.04 LTS, type `t3.large`, 30 GB gp3, a key pair for SSH.
- Security group: **80** and **443** from anywhere, **22** only from your own IP.
- Allocate an **Elastic IP** and attach it (so the address survives stop / start), then point your DNS name at it.
  For DuckDNS: set the IP on the duckdns.org page.

## 2. Prepare the machine (once)

```bash
ssh -i <key>.pem ubuntu@<elastic-ip>
git clone https://github.com/sachin-11/claimpilot.git && cd claimpilot
sudo bash deploy/server-setup.sh       # Docker + Compose, 4 GB swap
exit                                    # log in again so the docker group applies
```

## 3. Configure (once)

```bash
cd claimpilot
OPENAI_API_KEY=sk-... ./deploy/bootstrap-env.sh claimpilot-yourname.duckdns.org
```

This writes `.env` (mode 600) with random secrets and prints the admin and officer passwords **once**: save them.
The app refuses to start in production with default secrets, so there is nothing to forget.

Copy the policy PDF from your laptop (run this on the laptop):

```bash
scp -i <key>.pem data/policies/arogya_sanjeevani.pdf ubuntu@<elastic-ip>:claimpilot/data/policies/
```

## 4. Build and start

```bash
alias dc='docker compose -f docker-compose.prod.yml'
dc build backend frontend               # about 10-15 minutes the first time (torch for the reranker)
dc up -d
dc ps                                   # wait until postgres, redis, backend, worker, mcp-* are healthy
```

First start only:

```bash
dc exec backend alembic upgrade head
dc exec backend python -m app.scripts.seed                       # accounts + the 40 sample claims
dc exec backend python -m app.ingestion.run --policy POL-AS      # embeds the policy (OpenAI, ~1 minute)
```

The reranker model (~1 GB) downloads on the first policy question and is kept in the `claimpilot_hf_cache` volume. Ask
one question yourself before the demo so the first visitor does not wait.

## 5. Check it

1. `https://<your name>/api/health/ready` shows every dependency `ok` (database, Redis, the three MCP servers, the worker).
2. Log in as the officer, upload `data/claims/CLM-014/` (bill + discharge summary, policy `POL-AS/2024/7084652`,
   amount 71415), run it. Expected: partial approval, about Rs 53,666.
3. Log in as the admin (the officer uploaded the claim, so the officer cannot approve it) and approve.
4. Policy chat: "Is treatment abroad covered?" -> no, clause 7.19, high confidence.
5. Restart test: `sudo reboot`, wait a minute: the site comes back without any command.

## Stop, start, remove

| | |
|---|---|
| Pause (cheap) | EC2 console -> Stop. Everything stays on the disk. Start it again and the containers come back by themselves. |
| Update the code | `git pull && dc build backend frontend && dc up -d` |
| Look at logs | `dc logs -f backend worker` |
| Remove everything | Terminate the instance, release the Elastic IP, delete leftover volumes. Cost goes to $0. |

## Things to know

- **A public link with a real key behind it.** Anyone with an account can run claims. Share the logins only with the
  people who should see the demo; `DAILY_LLM_BUDGET_INR` (default 100 in the template) stops spending above the cap
  and `GET /admin/usage` shows it. There is no sign-up page.
- **The wording PDF** stays on the server (the citation drawer shows clause text to logged-in users, as in the app).
- **Notifications are mocked** (`sent_mock`): no e-mail leaves the server.
- **Memory.** If containers are killed (`dc ps` shows restarts, `dmesg | grep -i oom`), use `t3.large`.
- **Backups** are not set up. This is a demo: the seed script and the ingestion rebuild everything.
- Do not copy the AWS access keys from your laptop to this server. Nothing here needs them.
