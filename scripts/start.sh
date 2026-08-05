#!/usr/bin/env sh
# ===========================================================================
# SmartReco — container entrypoint for Railway / Render / Fly / any PaaS.
#
# Three phases, in order:
#   1. migrate  — bring the schema up to head
#   2. seed     — load the 50-course catalog + demo users (idempotent)
#   3. serve    — uvicorn bound to the platform-supplied $PORT
#
# Phases 1 and 2 are deliberately NON-FATAL. A managed database that is still
# accepting its first connection, or a transient Mesh hiccup during embedding,
# must not put the container into a crash-restart loop — the app creates any
# missing tables itself at startup and re-indexes the vector store on boot, so
# a failed migrate/seed degrades the demo rather than killing it.
# ===========================================================================

set -u

PORT="${PORT:-8000}"
AUTO_MIGRATE="${AUTO_MIGRATE:-true}"
AUTO_SEED="${AUTO_SEED:-true}"

# Resolve the interpreter rather than trusting PATH. Console scripts (`alembic`,
# `uvicorn`) are NOT reliably on PATH across build systems — a Nixpacks venv or a
# --user pip install puts them somewhere the shell may not search, and the image
# may expose `python3` but not `python`. Everything below therefore goes through
# `<interpreter> -m <module>`, which works identically everywhere.
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[start] FATAL: no python interpreter found on PATH"
  exit 1
fi

echo "=========================================================="
echo "[start] SmartReco booting"
echo "[start] interpreter=$(command -v ${PY}) ($(${PY} --version 2>&1))"
echo "[start] port=${PORT} environment=${ENVIRONMENT:-development}"
echo "[start] migrate=${AUTO_MIGRATE} seed=${AUTO_SEED}"
echo "=========================================================="

# --- 1. Schema -------------------------------------------------------------
if [ "${AUTO_MIGRATE}" = "true" ]; then
  echo "[start] applying migrations (alembic upgrade head)..."
  if "${PY}" -m alembic upgrade head; then
    echo "[start] migrations applied"
  else
    echo "[start] WARNING: alembic failed — the app will create missing tables at startup"
  fi
else
  echo "[start] migrations skipped (AUTO_MIGRATE=false)"
fi

# --- 2. Catalog ------------------------------------------------------------
# Idempotent: courses are matched on title, so redeploys update in place rather
# than duplicating. Demo users and their synthetic session are created once.
if [ "${AUTO_SEED}" = "true" ]; then
  echo "[start] seeding catalog + demo users..."
  if "${PY}" -m scripts.seed_products --demo-users --with-events; then
    echo "[start] catalog seeded"
  else
    echo "[start] WARNING: seeding failed — the app will still serve; re-run from the admin UI"
  fi
else
  echo "[start] seeding skipped (AUTO_SEED=false)"
fi

# --- 3. Serve --------------------------------------------------------------
# Single worker is REQUIRED, not a default:
#   * APScheduler is in-process, so N workers would fire the daily digest N times
#   * the embedded Qdrant fallback lives in process memory and is not shared
# Scale out by adding a real Qdrant service and an external scheduler first.
#
# --proxy-headers + --forwarded-allow-ips are needed behind the platform's TLS
# terminator so the app sees the original https scheme and client IP.
echo "[start] launching uvicorn on 0.0.0.0:${PORT}"
exec "${PY}" -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips='*' \
  --timeout-keep-alive 65
