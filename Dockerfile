# ===========================================================================
# SmartReco — application image.
#
# Single-stage on purpose. Every dependency in requirements.txt ships a
# manylinux wheel (psycopg2-binary bundles libpq, cryptography and bcrypt are
# prebuilt), so there is nothing to compile: no toolchain, no apt packages, and
# no separate wheel-building stage whose `--no-index` install can fail hard on a
# PaaS builder. Fewer moving parts, faster and more reliable cold builds.
#
# The container listens on $PORT so it works unmodified on Railway, Render, Fly
# and Cloud Run, all of which inject the port rather than letting you pick it.
# ===========================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/srv \
    PORT=8000

WORKDIR /srv

# Dependencies first, so the layer caches across code-only changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts
COPY app ./app

RUN chmod +x scripts/start.sh \
    && useradd --create-home --shell /bin/bash --uid 10001 smartreco \
    && chown -R smartreco:smartreco /srv
USER smartreco

# Documentation only — the platform maps its own port to $PORT.
EXPOSE 8000

# No container HEALTHCHECK: Railway/Render probe over HTTP themselves (see
# `healthcheckPath` in railway.json), and a curl-based check would mean adding
# apt packages purely to duplicate that.

CMD ["sh", "scripts/start.sh"]
