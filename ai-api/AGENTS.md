# QuickPitik - AI API

## Project Overview

This is a FastAPI (Python 3.11+) service for **blur detection**, called directly by the BatchMyPhotos desktop app. Optional C++ acceleration via pybind11 — dev-only: **no deployment image compiles the extension** (`.dockerignore` drops `*.pyd`; the Python fallbacks are the production path).

Face search and bib recognition were removed on 2026-10-16: the Spring Boot backend uses AWS Rekognition (`AI_PROVIDER=rekognition`) for both. The `persons` and `face_embeddings` tables (and their migrations) still exist in the database; no served code reads them.

- **Blur classifier** — two-stage (whole-frame rule, then person detector + per-person sharpness + logistic rule), four classes reported; person detector at `models/person_detector/yolov8n_seg_person.onnx`, rule at `models/blur_classifier/people_rule.json`; quality evidence and limitations are in `docs/ai-system-overview.md`

See `docs/ai-system-overview.md` for the full picture.

## Key Documents

| Document | Purpose |
|----------|---------|
| `AGENTS.md` | This file — entry point for AI agents and new team members |
| `docs/ai-system-overview.md` | Current state of the blur pipeline — models, accuracy, artifacts, endpoints |
| `docs/api-reference.md` | Every endpoint with request/response examples |
| `docs/architecture.md` | 4-layer separation, model registry, batch job flow |
| `docs/folder-structure.md` | What lives where |

## Architecture

**4-layer separation. Never skip layers.**

```
src/api/       → HTTP controllers (thin)
src/services/  → Business logic (none today; create it when a handler's logic grows)
src/ml/        → ML model wrappers (no HTTP, no DB awareness)
src/db/        → Database models, repositories, session management
```

- `api/` calls `services/` or `ml/` + `db/` directly. Never import `db/` directly from `api/` without going through a repository.
- `ml/` modules must never import from `api/`, `services/`, or `db/`.
- `schemas/` (Pydantic models) are shared across layers for request/response types.

Today there is no `src/services/` package: the blur endpoints call their ML wrappers (from the registry) and repositories directly from the API handler — a service class can be introduced later if the logic grows.

## Key Patterns

### Model Registry

All ML models are loaded **once** at startup via `src/ml/registry.py` and stored in `app.state.model_registry`. Never instantiate models inside route handlers or services — always pull from the registry:

```python
registry = request.app.state.model_registry
detector = registry.get("blur")           # BlurDetector
classifier = registry.get("blur_classifier")  # BlurClassifier, required in blur profile
```

`blur` is required in every profile; `blur_classifier` is required and verified in the `blur` profile (the default) and optional in the `all` profile. Both profiles load only these two models.

### API Response Envelope

Every endpoint returns `src/schemas/common.APIResponse`:

```python
return APIResponse(
    success=True,
    request_id=getattr(request.state, "request_id", ""),
    data=result.model_dump(),
)
```

Errors set `success=False` and populate `error={"code": ..., "message": ...}`.

### C++ Fallback Pattern

Any code that uses the C++ extension must use the try/except import pattern:

```python
try:
    from _quickpitik_cpp import some_function
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
```

Then branch on `_HAS_CPP` only where a function has the same output contract. Pure Python/NumPy fallback must always exist. Functions currently exposed by `_quickpitik_cpp` (see `src/cpp/bindings.cpp`): `laplacian_variance`, `fft_hf_ratio`, `batch_blur_metrics`, `bgr_to_gray`, `resize_gray`, `classify_preprocess`. The module also exports the `BlurMetrics` result struct. Nothing in `src/` imports the extension today; these functions remain exported but are not used by served inference: the blur classifier no longer calls `classify_preprocess` (it measures people at the 1280-px upload), and the C++ blur score differs from OpenCV at image borders.

### Database Access

Use async SQLAlchemy sessions via `src/db/session.get_session_ctx()`. Repositories (jobs, webhooks) handle CRUD.

Celery workers cannot use asyncpg — they use the sync engine in `src/db/sync_session.py` and the `sync_*_repo.py` repositories.

### Authentication + Rate Limiting

Every endpoint except `GET /health` and the minimal public `GET /health/readyz` uses `Depends(verify_api_key)` (`src/middleware/auth.py`). The dependency:

1. SHA-256-hashes the `X-API-Key` header
2. Checks Redis cache, then falls back to the `api_keys` table
3. Enforces the token-bucket rate limit for the key's tier (free=60/min, pro=300/min, internal=1000/min)
4. Stores rate info on `request.state.rate_info` so `RateLimitHeadersMiddleware` can set `X-RateLimit-*` response headers

In `DEBUG=true` mode a missing key is allowed and granted full scope (`["*"]`). Startup refuses `DEBUG=true` unless `ENVIRONMENT` is set to `development`, `local` or `test`.

Scopes used by the current code: `blur:read`, `jobs:read`, `webhooks:read`, `webhooks:write`.

### Batch Job Flow (blob-store, not base64)

Image bytes are never put on the Celery message queue. The pattern is:

1. API handler validates uploads with `validate_batch_files` (raw bytes, in-memory).
2. `create_batch_job` creates the job row and applies backpressure (`MAX_ACTIVE_JOBS_PER_KEY`, default 10 → 429 when exceeded).
3. `store_blobs_and_get_paths` atomically writes each image to `{BLOB_STORE_PATH}/{job_id}/NNNNN.bin` (default `/tmp/quickpitik-blobs`).
4. Celery task receives `(job_id, image_paths, ...)` — small payload.
5. Worker decodes from paths in parallel, runs inference in sub-batches of `INFERENCE_SUB_BATCH_SIZE` (50).
6. `complete_job` / `fail_job` updates the DB and fires any webhook subscriptions for `job.completed` / `job.failed`. It does **not** delete the staged images: they are kept as a retry window.
7. The beat task `cleanup_stale_blobs` (every 30 min, `src/workers/tasks/maintenance_tasks.py`) deletes a job's staged images once the job is `completed` or `failed` and finished at least `BLOB_RETENTION_SECONDS` ago (default 86400 = 24 h). A staged batch with no job row is deleted once its files are that old. Inputs of pending or processing jobs are never deleted.

For client-side pagination of large result sets, `GET /jobs/{id}` supports `offset` and `limit` query params.

### Three Batch Modes

| Mode | Max size | Returns | When to use |
|---|---|---|---|
| `.../batch` | `MAX_BATCH_SIZE` (50) | 202 + job_id (poll) | Normal async batches |
| `.../mega` | `MEGA_BATCH_MAX_SIZE` (500) | 202 + job_id (poll) | Large uploads; server splits into sub-tasks via Celery chord |
| `.../stream` (blur only) | `STREAM_BATCH_MAX_SIZE` (500) | NDJSON stream, sync | High-throughput synchronous processing from desktop |

## File Locations

| To add... | Put it in... |
|---|---|
| New API endpoint | `src/api/v1/<feature>.py` then register in `src/api/v1/router.py` |
| New request/response model | `src/schemas/<feature>.py` |
| New business logic (if it grows) | `src/services/<feature>_service.py` |
| New ML model wrapper | `src/ml/<feature>/` with `__init__.py` |
| New database table | `src/db/models.py` then create Alembic migration |
| New repository (CRUD) | `src/db/repositories/<feature>_repo.py` (plus `sync_*_repo.py` for Celery) |
| New Celery task | `src/workers/tasks/<feature>_tasks.py` |
| New middleware | `src/middleware/<name>.py` then register in `src/main.py` |
| New config variable | Add to `src/config.py` Settings class and `.env.example` |

## Conventions

- **Python 3.11+** features are allowed (`type | None`, `list[str]`, `match` statements). The package supports 3.11–3.14.
- Use `from __future__ import annotations` at the top of every module.
- All route handlers are `async`. CPU-bound ML inference runs via `asyncio.to_thread()` when called from async context.
- API endpoints are versioned: `/api/v1/...`. Future breaking changes go in `src/api/v2/` (not yet created).
- Logging: use `from src.utils.logging import get_logger; logger = get_logger(__name__)`. Always structured (key=value), never f-string log messages.
- Exceptions: use custom types from `src/utils/exceptions.py` (`QuickPitikError` and subclasses). The `QuickPitikError` handler in `main.py` turns them into the standard error envelope.
- Image validation: always call `validate_and_decode()` from `src/utils/image_utils.py` before processing uploads. Never trust Content-Type headers alone.
- Database: use `UUID` primary keys everywhere. Timestamps use `DateTime(timezone=True)`.
- Environment config: never hardcode values. Add to `src/config.py` and read from env vars.

## Running

```bash
# Install
pip install -e ".[dev]"

# Start infrastructure
docker compose up db redis -d

# Run migrations
alembic upgrade head

# Seed dev data (creates an API key)
python scripts/seed_db.py

# Start dev server
make dev

# Start Celery worker (separate shell)
make celery

# Run tests
make test

# Lint
make lint
```

## Testing

- Tests live in three places:
  - `tests/` (top level) — the primary suite, e.g. `test_blur_detector.py`, `test_blur_classifier.py`, `test_blur_endpoint.py`, `test_batch_endpoints.py`, `test_cpp_extension.py`, `test_deployment_config.py`.
  - `tests/unit/`, `tests/integration/`, `tests/e2e/` — additional unit / integration / end-to-end scoped tests.
- Test fixtures (images) go in `tests/fixtures/`.
- Shared fixtures are in `tests/conftest.py`.
- Use `pytest-asyncio` for async tests. Config: `asyncio_mode = "auto"` in pyproject.toml.

## Dependencies

- **Web**: FastAPI, Uvicorn, Pydantic v2, python-multipart
- **Image**: OpenCV (headless), NumPy, Pillow, pillow-heif
- **ML**: ONNX Runtime; Ultralytics only in the optional `train` extra (`pip install -e ".[train]"`) for the offline blur export/training scripts
- **DB**: PostgreSQL 16 + pgvector, SQLAlchemy 2 (async + sync), Alembic, asyncpg, psycopg2
- **Queue**: Celery 5 + Redis 7
- **Auth**: API keys (SHA-256 hashed), PyJWT (with `cryptography`), bcrypt
- **Observability**: structlog, prometheus-client, prometheus-fastapi-instrumentator
- **C++ (optional)**: pybind11, scikit-build-core, CMake, Ninja

Do not add new dependencies without justification. Prefer existing libraries over new ones.

## Things to Avoid

- Never load ML models inside request handlers. Use the registry.
- Never store uploaded images to disk or database beyond the short-lived blob staging directory used for Celery batches.
- Never log image data or embeddings. Log only request IDs, endpoints, and timings.
- Never commit `.env` files, API keys, or the `WEBHOOK_SECRET_KEY`.
- Never import across layers incorrectly (e.g., `api/` importing from `db/` without going through a repository).
- Never use synchronous database calls in FastAPI request handlers. Always use async SQLAlchemy. (Exception: Celery workers **must** use sync sessions via `src/db/sync_session.py` because asyncpg cannot run inside Celery.)
- Never hardcode thresholds, URLs, or secrets. Use `src/config.py`.

## ML Features

### Blur Detection

Two systems coexist:

1. **Laplacian/FFT detector** (`src/ml/blur/detector.py`): Always-available classical CV. Single, stream, and worker decode grayscale at `BLUR_DETECT_DECODE_DIM` and use the same OpenCV Laplacian score, normalized to a 640px reference. FFT is informational and runs only when requested. `detect_fast(gray)` skips FFT.
2. **Two-stage blur classifier** (`src/ml/blur/classifier.py`): whole-frame rule (`src/ml/blur/whole_frame.py`; thresholds `BLUR_SOFT_MAX_DETAIL`, `BLUR_MOTION_MIN_RATIO`, `BLUR_MOTION_MAX_DETAIL`), then person detector + per-person sharpness + logistic rule (`src/ml/blur/person_detector.py`, `src/ml/blur/people.py`); four classes reported (sharp, defocused_object_portrait, defocused_blurred, motion_blurred). Supports GPU via `USE_GPU`. Targeted detection enforces a minimum confidence floor (`BLUR_DETECTION_MIN_CONFIDENCE`, default 0.5). Requires the person detector at `models/person_detector/yolov8n_seg_person.onnx` and the rule at `models/blur_classifier/people_rule.json`. Required and verified in the `blur` profile; optional in `all`.

API endpoints:

- `POST /api/v1/blur/detect` — single image, Laplacian/FFT (always available)
- `POST /api/v1/blur/detect/stream` — NDJSON stream, up to 500 images, synchronous
- `POST /api/v1/blur/detect/batch` — async Celery, up to `MAX_BATCH_SIZE` (50)
- `POST /api/v1/blur/detect/mega` — async Celery chord, up to 500 images
- `POST /api/v1/blur/classify` — single image, two-stage classifier (optional `blur_type` query param)
- `POST /api/v1/blur/classify/stream` — NDJSON stream, up to 500
- `POST /api/v1/blur/classify/batch` — async Celery, up to 50
- `POST /api/v1/blur/classify/mega` — async Celery chord, up to 500

## Async Batch Processing

Celery configuration lives in `src/workers/celery_app.py`. Highlights:

- **Broker & backend**: Redis (same `REDIS_URL`)
- **Queues**: `default`, `blur` — tasks routed via `task_routes`
- **Time limits**: soft `task_soft_time_limit=3300s`, hard `task_time_limit=3600s`
- **Worker protection**: `worker_max_tasks_per_child=500`, `worker_max_memory_per_child=2GB`
- **Prefetch**: `worker_prefetch_multiplier=4` (safe because task messages carry file paths, not image bytes)
- **Beat schedule**: `reap_stale_jobs` (5 min), `cleanup_old_jobs` (daily, `JOB_RETENTION_DAYS=7`), `cleanup_stale_blobs` (30 min)
- **Message signing**: JSON serializer; `CELERY_SECURITY_KEY` is accepted but not yet used (proper X.509 PKI required for the `auth` serializer)

Per-queue worker deployment (production):

```bash
# Workers can load only what they need via WORKER_QUEUES
WORKER_QUEUES=blur    celery -A src.workers.celery_app worker -Q blur -c 4
WORKER_QUEUES=default celery -A src.workers.celery_app worker -Q default -c 2
```

A `default` worker (webhooks, maintenance) loads no models. When `WORKER_QUEUES` is unset (dev) it means `blur,default`, so the worker loads the blur models.

Key worker files:

- `src/workers/celery_app.py` — config, beat schedule, task routes
- `src/workers/model_loader.py` — queue-aware model loading via `worker_process_init`
- `src/workers/helpers.py` — decode helpers (path → grayscale / BGR), job progress/complete/fail, webhook dispatch
- `src/workers/tasks/` — `blur_tasks`, `webhook_tasks`, `maintenance_tasks`

## Model Retraining

Training scripts live in `scripts/`:

- Blur classifier: `build_blur_folds.py` → `fit_blur_whole_frame_rule.py` → `export_person_detector.py` → `measure_blur_people.py` → `fit_blur_people_rule.py` → `ship_blur_rule.py`

`export_person_detector.py` needs the `train` extra (`pip install -e ".[train]"`). Training images live in `Training-Images/` (gitignored). Model artifacts live in `models/` (gitignored except `manifest.json`). See `docs/ai-system-overview.md` for the current state of the pipeline.

## Infrastructure Requirements

| Service | Purpose | Required for |
|---------|---------|-------------|
| PostgreSQL 16 + pgvector | Jobs, webhook subscriptions, API keys (the retired `persons` / `face_embeddings` tables still exist; pgvector is needed by their migrations) | All features |
| Redis 7 | Celery broker/backend, rate limiting, API key cache | All features (authenticated requests return 503 while Redis is unavailable) |
| Celery worker | Async task execution | Batch / mega endpoints and webhooks |
| Blob store volume | `BLOB_STORE_PATH` (default `/tmp/quickpitik-blobs`) — shared between API and workers | Batch / mega endpoints |

Start infra: `docker compose up db redis -d`
