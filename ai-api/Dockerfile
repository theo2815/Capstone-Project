# ---- Stage 1: Python dependencies ----
FROM python:3.12-slim AS builder

WORKDIR /build

# psycopg2 (not psycopg2-binary) publishes no Linux wheels, so pip builds it
# from source here. Without these the build dies at
# "Error: pg_config executable not found" — and with libpq-dev but no libc
# headers, at "fatal error: stdlib.h: No such file or directory". This is the
# reason the production image had never been built. All of it stays in the
# builder stage and is discarded; runtime keeps only libpq5.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime

# libpq5 is the runtime half of the psycopg2 source build above — the compiled
# extension links against it, so importing psycopg2 fails without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libpq5 curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

COPY src/ /app/src/
# Manifest only — the ~618 MB of weights are mounted read-only from the host at
# /app/models by docker-compose.prod.yml. They cannot be baked: models/* is
# gitignored, so a build from a clean checkout has nothing to copy. Copying the
# whole directory here looked like it delivered them and delivered nothing,
# because .dockerignore excludes it.
COPY models/manifest.json /app/models/manifest.json
COPY alembic.ini /app/alembic.ini

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Non-root user for security
RUN adduser --disabled-password --no-create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-graceful-shutdown", "30"]
