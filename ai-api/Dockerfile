# ---- Stage 1: Python dependencies ----
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS builder
ARG MODEL_PROFILE=all

WORKDIR /build

# psycopg2 (not psycopg2-binary) publishes no Linux wheels, so pip builds it
# from source here. Without these the build dies at
# "Error: pg_config executable not found" — and with libpq-dev but no libc
# headers, at "fatal error: stdlib.h: No such file or directory". This is the
# reason the production image had never been built. All of it stays in the
# builder stage and is discarded; runtime keeps only libpq5.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements-blur.lock ./
RUN case "$MODEL_PROFILE" in \
      blur) pip install --no-cache-dir --prefix=/install -r requirements-blur.lock ;; \
      all) pip install --no-cache-dir --timeout 120 --prefix=/install . ;; \
      *) echo "Invalid MODEL_PROFILE: $MODEL_PROFILE" >&2; exit 1 ;; \
    esac

# ---- Stage 2: Runtime ----
FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9 AS runtime
ARG MODEL_PROFILE=all
ENV MODEL_PROFILE=$MODEL_PROFILE
ENV HOME=/app

# libpq5 is the runtime half of the psycopg2 source build above — the compiled
# extension links against it, so importing psycopg2 fails without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libpq5 curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

COPY src/ /app/src/
# The audited blur pair is supplied to the build context, never committed.
# .dockerignore admits only the rule file, the 13 MB person detector and the manifest. Full-profile
# deployments can still mount their other models as before.
COPY models/ /app/models/
COPY alembic.ini /app/alembic.ini

RUN if [ "$MODEL_PROFILE" = "blur" ]; then \
      python -c 'import sys; from src.ml.blur.classifier import BlurClassifier; c = BlurClassifier(manifest_path="/app/models/manifest.json"); sys.exit(0 if c.session is not None else "Verified blur artifact is unavailable")'; \
    fi

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Non-root user for security
RUN adduser --disabled-password --no-create-home appuser && \
    chown -R appuser:appuser /app && chmod -R a-w /app/models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/v1/health/readyz" || exit 1

CMD ["sh", "-c", "if [ -z \"${WORKERS:-}\" ]; then if [ \"$MODEL_PROFILE\" = blur ]; then WORKERS=1; else WORKERS=2; fi; fi; exec uvicorn src.main:create_app --factory --host 0.0.0.0 --port \"${PORT:-8000}\" --workers \"$WORKERS\" --timeout-graceful-shutdown 30"]
