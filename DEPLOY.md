# Deploying SmartReco to Railway

Target: a live public URL in ~10 minutes, with a working demo (38 courses, two
demo accounts, a pre-seeded browsing session and a real AI recommendation).

---

## What the platform runs

| | |
|---|---|
| **Entrypoint (ASGI app)** | `app.main:app` |
| **Build** | None required — the `Dockerfile` runs `pip install -r requirements.txt`. Railway auto-detects it. |
| **Start command** | `sh scripts/start.sh` (already the Dockerfile `CMD` and `railway.json` `startCommand`) |
| **Port** | The container binds `0.0.0.0:$PORT`. Railway injects `PORT`; never hardcode it. |
| **Health check** | `GET /health` → `{"status":"ok"}` (wired via `railway.json`) |
| **Workers** | Exactly **1** — see [Why one worker](#why-one-worker) |

`scripts/start.sh` runs three phases, the first two deliberately non-fatal so a
slow database or a transient Mesh hiccup degrades the demo instead of crash-looping:

1. `python -m alembic upgrade head` — create the schema
2. `python -m scripts.seed_products --demo-users --with-events` — idempotent catalog seed
3. `exec python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --proxy-headers`

Set `AUTO_MIGRATE=false` or `AUTO_SEED=false` to skip either phase.

---

## Environment variables

### Required

| Variable | Value | Why |
|---|---|---|
| `MESH_API_KEY` | your `rsk_…` key | Without it the agent still runs, but in degraded mode: templated copy instead of written copy. **Set this or the demo undersells itself.** |
| `SECRET_KEY` | 32+ random chars | Signs JWTs and flash cookies. Generate: `python -c "import secrets;print(secrets.token_urlsafe(48))"` |

### Strongly recommended

| Variable | Value | Why |
|---|---|---|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` | Railway reference variable. Without it the app uses SQLite on an **ephemeral disk** — every redeploy wipes accounts and recommendations. |
| `ENVIRONMENT` | `production` | Hardens cookies to `Secure` (correct — Railway terminates TLS). |
| `DEBUG` | `false` | Stops tracebacks rendering in error pages. |
| `BASE_URL` | `https://<your-app>.up.railway.app` | Used for links in digest emails. |

### Optional

| Variable | Value | Effect |
|---|---|---|
| `REDIS_URL` | `${{Redis.REDIS_URL}}` | Shared locks + result cache. Omit and the app uses an in-process fallback with identical semantics — fine at one replica. |
| `QDRANT_URL` | `http://<qdrant-service>:6333` | A real vector server. Omit and the app uses an embedded in-memory index, re-built from SQL on every boot. |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | `true` / your key | Full graph tracing to project `smartreco-agent`. |
| `EMAIL_BACKEND` | `console` (default) / `sendgrid` / `smtp` | `console` logs digests instead of sending — safe by default. |
| `SCHEDULER_ENABLED` | `true` (default) | Daily digest + background agent queue. Safe to leave on. |
| `AUTO_SEED` | `true` (default) | Set `false` after the first successful deploy if you want deploys to stop touching the catalog. |

`.env` is **not** used on Railway — variables come from the dashboard. Never commit `.env`.

---

## Do I need PostgreSQL, Redis and Qdrant?

**PostgreSQL — recommended, not required.** The app runs on SQLite out of the box,
but Railway's filesystem is ephemeral: registered users, tracked events and
generated recommendations vanish on every redeploy. It's one click to add, so add it.

**Redis — genuinely optional.** It provides the per-user agent lock and the
recommendation result cache. With one replica, the in-process fallback is
functionally equivalent. Add it only when you scale past one instance.

**Qdrant — optional for a demo.** The embedded fallback runs real cosine ANN
search in process and is re-indexed from SQL at every boot, so retrieval is fully
functional; it just doesn't persist and isn't shared. To add a real one: **New →
Docker Image → `qdrant/qdrant`**, then set `QDRANT_URL` to its private URL.

---

## Why one worker

Two components are process-local by design:

- **APScheduler** runs in-process, so *N* workers would fire the daily digest *N* times.
- **The embedded Qdrant fallback** lives in process memory and is not shared between workers.

`scripts/start.sh` therefore pins `--workers 1`. To scale horizontally, first add
a real Qdrant service and move the scheduler to a separate Railway service
(same image, `SCHEDULER_ENABLED=true` there and `false` on the web service).

---

## Dashboard steps

1. **New Project → Deploy from GitHub repo** → pick your repo. Railway detects the
   `Dockerfile` and starts building; the first build takes ~3–5 minutes.
2. **Add PostgreSQL**: *New → Database → Add PostgreSQL*.
3. **Variables** on the app service — add:
   ```
   MESH_API_KEY   = rsk_...
   SECRET_KEY     = <random 32+ chars>
   DATABASE_URL   = ${{Postgres.DATABASE_URL}}
   ENVIRONMENT    = production
   DEBUG          = false
   ```
   `${{Postgres.DATABASE_URL}}` is a reference variable — type it literally and
   Railway resolves it.
4. **Generate a domain**: *Settings → Networking → Generate Domain*. Railway
   detects the exposed port automatically.
5. Add `BASE_URL = https://<that-domain>` and let it redeploy.
6. **Verify**:
   - `https://<domain>/health` → `{"status":"ok"}`
   - `https://<domain>/health/ready` → all checks `true`, `mesh_configured: true`
   - `https://<domain>/` → homepage
7. **Demo it**: sign in as `learner@smartreco.ai` / `smartreco123`. The seeded
   session triggers the agent on first visit (`first_time`), so a recommendation
   appears within seconds — or press **Refresh my picks**. Then open
   **Why this recommendation?** to show the interest signals and evidence.
   `admin@smartreco.ai` / `smartreco123` gets the dual-write dashboard.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build succeeds, health check fails | App not on `$PORT` | Confirm the start command is `sh scripts/start.sh`; don't override it with a hardcoded `--port` |
| `NoSuchModuleError: postgres` | Bare `postgres://` DSN | Already handled — `app/config.py` rewrites it to `postgresql+psycopg2://` |
| Catalog is empty | Seed phase failed | Check deploy logs for `[start] WARNING: seeding failed`; or hit **Re-index catalog** in `/admin` |
| Recommendation says "degraded mode" | `MESH_API_KEY` missing or invalid | Set it in Variables; verify at `/health/ready` → `mesh_configured: true` |
| Data disappears after redeploy | SQLite on ephemeral disk | Add PostgreSQL and set `DATABASE_URL` |
| Login redirects back to `/login` | `ENVIRONMENT=production` over plain HTTP | Use the `https://` domain — production cookies are `Secure` |
