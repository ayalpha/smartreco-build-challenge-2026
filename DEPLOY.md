# Deploying Nexora

Two verified targets. **Render** is documented first because its free tier needs
no add-on services at all. Railway follows.

Common to both: the container binds `0.0.0.0:$PORT` (the platform injects it) and
`scripts/start.sh` migrates → seeds → serves.

| | |
|---|---|
| **Entrypoint (ASGI app)** | `app.main:app` |
| **Start command** | `sh scripts/start.sh` |
| **Health check path** | `/health` |
| **Workers** | exactly **1** — see [Why one worker](#why-one-worker) |
| **Measured memory** | ~170 MB resident (fits a 512 MB free instance) |

---

# Render (free tier)

## Recommended shape

| Setting | Value | Why |
|---|---|---|
| Service type | **Web Service** | |
| Runtime | **Docker** | Guarantees Python 3.11. Render's native Python default drifts between versions and this project needs 3.11+. The Dockerfile is already `$PORT`-aware. |
| Instance type | **Free** | 512 MB / 0.1 CPU. Measured footprint ~170 MB, so there's headroom. |
| Build command | *(none — Docker handles it)* | |
| Start command | *(none — the Dockerfile `CMD` runs `sh scripts/start.sh`)* | |
| Health check path | `/health` | |
| Database | **SQLite first** | Zero add-ons. See [Why SQLite is fine](#why-sqlite-is-fine-here). |
| Redis / Qdrant | **Skip both** | Built-in fallbacks cover them. |

Prefer the native Python runtime instead? It works — Build Command
`pip install -r requirements.txt`, Start Command `sh scripts/start.sh`. Keep
`runtime.txt` (pins 3.11.9) so Render doesn't pick a newer Python. Docker is
still the safer default.

## Dashboard steps

1. **New + → Web Service** → **Build and deploy from a Git repository** → connect
   GitHub → pick your repo.
2. Render reads the `Dockerfile` and preselects **Docker**. Leave Build Command
   and Start Command **blank**.
3. **Instance Type → Free.**
4. **Advanced → Health Check Path** → `/health`.
5. **Environment Variables** — add these two:

   | Key | Value |
   |---|---|
   | `MESH_API_KEY` | your `rsk_…` key (paste it here, never into the repo) |
   | `SECRET_KEY` | click **Generate** |

   Then these four:

   | Key | Value |
   |---|---|
   | `ENVIRONMENT` | `production` |
   | `DEBUG` | `false` |
   | `DATABASE_URL` | `sqlite:///./smartreco.db` |
   | `EMAIL_BACKEND` | `console` |

   Do **not** set `BASE_URL` — `app/config.py` reads Render's
   `RENDER_EXTERNAL_URL` automatically.
6. **Create Web Service.** First build ~4–6 minutes.
7. **Verify** at `https://<your-service>.onrender.com`:
   - `/health` → `{"status":"ok"}`
   - `/health/ready` → all checks `true`, `mesh_configured: true`
   - `/` → homepage
8. **Demo the flow**: sign in as `learner@smartreco.ai` / `smartreco123`. The
   seeded browsing session triggers the agent on first visit (`first_time`), so a
   recommendation appears within seconds — or press **Refresh my picks**. Expand
   **Why this recommendation?** to show the interest signals and evidence.
   `admin@smartreco.ai` / `smartreco123` opens the dual-write dashboard.

Or skip 1–6: **New + → Blueprint** and point at the repo — `render.yaml` sets all
of the above, generates `SECRET_KEY`, and prompts only for `MESH_API_KEY`.

## Free-tier behaviour you should expect

- **Sleeps after ~15 min idle**, cold start ~1 minute. The first request after a
  sleep is slow; everything after it is fast. Fine for judging — send judges to
  `/health` first to wake it, then the homepage.
- **Ephemeral disk.** Data resets on every deploy and on wake-from-sleep.
- **The daily 18:00 digest will usually not fire**, because the instance is
  asleep at that hour. Not a defect and nothing to disable — demo the digest
  on demand with **Run digest now** on `/admin`.

## Why SQLite is fine here

Normally an ephemeral disk plus SQLite means "your demo data disappears". Here it
doesn't matter, because `scripts/start.sh` re-seeds on every boot: 38 courses, both
demo accounts, and a synthetic browsing session weighted toward Agentic AI. A wipe
self-heals into exactly the state you want a judge to land on.

What you lose: accounts *judges* create, and recommendations generated before a
restart. For a hackathon demo that's a non-issue.

Want persistence anyway? Add **New + → Postgres** (free plan), copy its **Internal
Database URL**, and set `DATABASE_URL` to it. The app normalises the DSN scheme
itself. Two caveats: Render's free Postgres **expires after 30 days**, and it adds
a service you have to keep alive. That's why SQLite is the default recommendation.

## Can Redis and Qdrant really be skipped?

**Yes — both, by design.**

| Service | Fallback | What you lose |
|---|---|---|
| Redis | In-process cache with identical TTL and lock semantics | Nothing at 1 instance. It only matters across replicas. |
| Qdrant | Embedded in-memory index, re-built from SQL at every startup | Persistence and sharing. Retrieval itself is fully functional — real cosine ANN, hybrid BM25 fusion, metadata filters. |

Leave `REDIS_URL` and `QDRANT_URL` empty (or unset). The startup logs will say
`using embedded in-memory Qdrant` and `REDIS_URL not set — using in-memory cache`;
both are expected. `/health/ready` reports `vector_store_mode: embedded`.

---

# Railway

Same container, same start script. Railway injects `PORT` and provides one-click
Postgres and Redis.

1. **New Project → Deploy from GitHub repo.** Railway detects the `Dockerfile`.
2. **New → Database → Add PostgreSQL** (recommended here, since Railway's disk is
   also ephemeral and its Postgres has no expiry).
3. **Variables:**
   ```
   MESH_API_KEY = rsk_...
   SECRET_KEY   = <32+ random chars>
   DATABASE_URL = ${{Postgres.DATABASE_URL}}
   ENVIRONMENT  = production
   DEBUG        = false
   ```
   `${{Postgres.DATABASE_URL}}` is a reference variable — type it literally.
4. **Settings → Networking → Generate Domain.**
5. `BASE_URL` is optional — the app reads `RAILWAY_PUBLIC_DOMAIN` automatically.

`railway.json` pins the Dockerfile builder, the `/health` check and 1 replica.

---

# Reference

## Environment variables

### Required

| Variable | Value |
|---|---|
| `MESH_API_KEY` | your `rsk_…` key. Without it the agent still runs, but in degraded mode: templated copy instead of written copy. |
| `SECRET_KEY` | 32+ random chars. Signs JWTs and flash cookies. `python -c "import secrets;print(secrets.token_urlsafe(48))"` |

### Recommended

| Variable | Value | Effect |
|---|---|---|
| `ENVIRONMENT` | `production` | Hardens cookies to `Secure` — correct, both platforms serve HTTPS. |
| `DEBUG` | `false` | Stops tracebacks rendering in error pages. |
| `DATABASE_URL` | `sqlite:///./smartreco.db` or a Postgres DSN | Bare `postgres://` and `postgresql://` are both normalised to `postgresql+psycopg2://`. |

### Optional

| Variable | Default | Effect |
|---|---|---|
| `BASE_URL` | auto-detected | Only for links in digest emails. Read from `RENDER_EXTERNAL_URL` / `RAILWAY_PUBLIC_DOMAIN`. |
| `REDIS_URL` | unset | Shared locks + cache. Empty ⇒ in-process fallback. |
| `QDRANT_URL` | unset | Real vector server. Empty ⇒ embedded index. |
| `EMAIL_BACKEND` | `console` | `console` logs digests instead of sending. |
| `SCHEDULER_ENABLED` | `true` | Daily digest **and** the background agent queue. Leave on. |
| `AUTO_SEED` | `true` | Re-seed on boot. Set `false` once you have a persistent database. |
| `AUTO_MIGRATE` | `true` | Run `alembic upgrade head` on boot. |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | off | Graph tracing to project `smartreco-agent`. |

`.env` is not used on either platform — variables come from the dashboard.

## Should APScheduler stay enabled?

**Yes, on both.** It does double duty: the daily digest cron *and* the background
queue that event-triggered agent runs are dispatched to. Disabling it doesn't break
recommendations — the runner falls back to its own thread pool — but there's no
reason to. On Render's free tier the digest simply won't fire while the instance
sleeps, which is harmless.

## Why one worker

- **APScheduler** is in-process, so *N* workers would fire the digest *N* times.
- **The embedded Qdrant index** lives in process memory and isn't shared.

`scripts/start.sh` pins `--workers 1`. To scale out, add a real Qdrant service and
move the scheduler to its own service (`SCHEDULER_ENABLED=true` there, `false` on web).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build ok, health check fails | App not on `$PORT` | Leave Start Command blank (Docker) or use `sh scripts/start.sh`. Never hardcode `--port`. |
| `NoSuchModuleError: postgres` | Bare `postgres://` DSN | Already handled in `app/config.py`. |
| Catalog empty | Seed phase failed | Look for `[start] WARNING: seeding failed` in logs; or hit **Re-index catalog** on `/admin`. |
| "degraded mode" badge | `MESH_API_KEY` missing/invalid | Set it; confirm `/health/ready` → `mesh_configured: true`. |
| Login bounces to `/login` | `ENVIRONMENT=production` over plain HTTP | Use the `https://` URL — production cookies are `Secure`. |
| First request takes ~1 min | Free-tier cold start | Expected. Hit `/health` to wake it before demoing. |
| Data gone after redeploy | Ephemeral disk + SQLite | Expected and self-healing; add Postgres if you need persistence. |
