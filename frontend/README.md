# ClaimPilot web app

React 19 + Vite 8 + TypeScript (strict) + Tailwind 4. Talks only to its own origin: `/api/*` is proxied to
the backend (nginx in the container, the Vite dev server locally), so the httpOnly refresh cookie is
same-site and there is no CORS.

## Run

```bash
# whole stack in Docker (app on http://localhost:3000)
docker compose up -d --build

# development with hot reload (backend must be running; BACKEND_HOST_PORT defaults to 8000, 8010 in this repo's .env)
cd frontend && npm install
VITE_BACKEND_URL=http://localhost:8010 npm run dev      # http://localhost:5173
```

No API keys / no cost: `make e2e-up` starts an isolated stack (own DB) with `MOCK_LLM=true` and demo data on
http://localhost:3300 (see below).

## Scripts

| | |
|---|---|
| `npm run dev` / `build` / `preview` | Vite |
| `npm run typecheck` / `npm run lint` | `tsc --noEmit`, ESLint (react-hooks rules included) |
| `npm test` | Vitest + Testing Library (unit and component tests) |
| `npm run gen:api` | regenerates `src/api/schema.d.ts` from the running backend (`OPENAPI_URL` to override) |
| `npm run e2e` | Playwright against `http://localhost:3300` (`make e2e-up` first) |

## Layout

```
src/
  api/        client.ts (openapi-fetch + silent refresh), sse.ts (authenticated SSE), upload.ts (XHR progress),
              types.ts (hand-written shapes the OpenAPI schema leaves open), schema.d.ts (GENERATED)
  stores/     auth.ts: zustand; the access token lives in memory only
  app/        App (routes, lazy pages), AppShell, guards (RequireAuth / RequireAdmin)
  components/ ui, overlays (Dialog / Drawer), tabs, toast, status chips
  features/   auth, claims (list, upload), claim-detail (viewer, tabs, approval), run (timeline + SSE hook),
              citations (chip + clause drawer), chat, admin
  lib/        format (INR, dates, durations), theme
e2e/          Playwright tests; `SHOTS=1 npm run e2e` also refreshes docs/screenshots/
```

## Decisions worth knowing

- **Auth:** access token in memory, refresh token in an httpOnly `SameSite=Strict` cookie scoped to `/api/auth`
  (`X-Forwarded-Prefix: /api` tells the backend the path). A 401 triggers ONE refresh (shared by parallel
  requests, because the refresh token rotates) and a retry; if the refresh fails the session is cleared and the
  route guard redirects to `/login`. Reloading the page restores the session from the cookie.
- **SSE:** `EventSource` cannot send an `Authorization` header, so `@microsoft/fetch-event-source` is used through
  the same authenticated `fetch`. Every (re)connection replays the run, so the timeline resets on open instead of
  appending duplicates. `features/run/timeline.ts` is a pure fold of events into rows (tested).
- **Permissions in the UI are a convenience:** the API enforces roles and separation of duties. The UI hides the
  Admin area from officers and locks the approval panel for the uploader, with the reason shown.
- **Citations:** every cited clause is a chip that opens a drawer with the full clause text from
  `GET /policies/chunks/{id}`; the same component serves the coverage table, the decision and the policy chat.
