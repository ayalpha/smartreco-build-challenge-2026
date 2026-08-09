#!/usr/bin/env bash
set -e

.venv/bin/python -m alembic upgrade head
exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
