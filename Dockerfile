# ===========================================================================
# SmartReco — application image.
#
# Multi-stage: wheels are built in a throwaway stage so the runtime image
# carries no compiler toolchain. Runs as an unprivileged user.
# ===========================================================================

# --------------------------------------------------------------- builder ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# psycopg2 and cryptography need a toolchain to build; only in this stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


# --------------------------------------------------------------- runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/srv

# libpq5 is the psycopg2 runtime dependency; curl is handy for probes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-index --find-links=/wheels -r /tmp/requirements.txt \
    && rm -rf /wheels /tmp/requirements.txt

WORKDIR /srv

# Application code. Ordered so dependency layers cache independently.
COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts
COPY app ./app

# Drop privileges.
RUN useradd --create-home --shell /bin/bash --uid 10001 smartreco \
    && chown -R smartreco:smartreco /srv
USER smartreco

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
