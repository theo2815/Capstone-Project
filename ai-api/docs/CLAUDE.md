# EventAI - AI API

## Project Overview

This is a modular AI API built with FastAPI (Python 3.11+) for computer vision tasks: **Blur Detection**, **Face Recognition**, and **Bib Number OCR**. Optional C++ acceleration via pybind11.

## Current Status

| Phase | Feature | Status |
|-------|---------|--------|
| Phase 1 | Foundation (FastAPI, DB, auth, middleware) | **Complete** |
| Phase 2 | Blur Detection (Laplacian-based) | **Complete** |
| Phase 3 | Face Recognition (InsightFace + pgvector) | **Complete** |
| Phase 4 | Bib Number Recognition (PaddleOCR) | **Complete** |
| Phase 5 | Async Batch Processing (Celery + Redis) | **Complete** |
| Phase 6 | C++ Acceleration (pybind11) | **Complete** |
| Phase 6.5 | API Abuse Prevention & Usage Control | Pending |
| Phase 7 | Production Hardening | Pending |

**Blur classifier training** is complete — see `docs/phase-plan-for-blur-detection-training.md` for full details. Round 3 achieved **98.68% accuracy** (56 epochs, early stopping). Sharp class: 100% (zero false positives). ONNX model exported to `models/blur_classifier/blur_classifier.onnx`. See `docs/phase-plan-face-bibnumber-training.md` for Phase 6 (Face Recognition) and Phase 7 (Bib Number) training plans.

## Key Documents

| Document | Purpose |
|----------|---------|
| `docs/CLAUDE.md` | This file — entry point for AI agents and new team members |
| `docs/phase-plan.md` | Full implementation phase plan with completed tasks and test results |
| `docs/phase-plan-for-blur-detection-training.md` | Blur classifier training plan, dataset details, blur detection logic rules, and accuracy targets |
| `docs/phase-plan-face-bibnumber-training.md` | Face recognition (Phase 6) and bib number (Phase 7) training plan — combined detection pipeline |

## Architecture

**4-layer separation. Never skip layers.**

```
src/api/       → HTTP controllers (thin, no logic)
src/services/  → Business logic and orchestration
src/ml/        → ML model wrappers (no HTTP, no DB awareness)
src/db/        → Database models, repositories, session management
```

- `api/` calls `services/`, services call `ml/` and `db/`. Never import `db/` directly from `api/`.
- `ml/` modules must never import from `api/`, `services/`, or `db/`.
- `schemas/` (Pydantic models) are shared across layers for request/response types.

## Key Patterns

### Model Registry

All ML models are loaded **once** at startup via `src/ml/registry.py` and stored in `app.state.model_registry`. Never instantiate models inside route handlers or services. Always get them from the registry:

```python
registry = request.app.state.model_registry
detector = registry.get("blur")
```

### API Response Envelope

Every endpoint returns `src/schemas/common.APIResponse`:

```python
return APIResponse(
    success=True,
    request_id=getattr(request.state, "request_id", ""),
    data=result.model_dump(),
)
```

On errors:
```python
return APIResponse(
    success=False,
    request_id=getattr(request.state, "request_id", ""),
    error={"code": "ERROR_CODE", "message": "Human-readable message"},
)
```

### C++ Fallback Pattern

Any code that uses the C++ extension must use the try/except import pattern:

```python
try:
    from _eventai_cpp import some_function
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
```

Then branch on `_HAS_CPP`. Pure Python/NumPy fallback must always exist.

### Database Access

Use async SQLAlchemy sessions via `src/db/session.get_session()`. Repositories handle CRUD. Face embedding search uses pgvector cosine distance (`<=>` operator).

### Authentication

All endpoints except health checks use `Depends(verify_api_key)` from `src/middleware/auth.py`. In debug mode, missing keys are allowed.

## File Locations

| To add... | Put it in... |
|---|---|
| New API endpoint | `src/api/v1/<feature>.py` then register in `src/api/v1/router.py` |
| New request/response model | `src/schemas/<feature>.py` |
| New business logic | `src/services/<feature>_service.py` |
| New ML model wrapper | `src/ml/<feature>/` with `__init__.py` |
| New database table | `src/db/models.py` then create Alembic migration |
| New repository (CRUD) | `src/db/repositories/<feature>_repo.py` |
| New Celery task | `src/workers/tasks/<feature>_tasks.py` |
| New middleware | `src/middleware/<name>.py` then register in `src/main.py` |
| New config variable | Add to `src/config.py` Settings class and `.env.example` |

## Conventions

- **Python 3.11+** features are allowed (`type | None`, `list[str]`, `match` statements).
- Use `from __future__ import annotations` at the top of every module.
- All route handlers are `async`. CPU-bound ML inference runs via `asyncio.to_thread()` when called from async context.
- API endpoints are versioned: `/api/v1/...`. Future breaking changes go in `src/api/v2/`.
- Logging: use `from src.utils.logging import get_logger; logger = get_logger(__name__)`. Always structured (key=value), never f-string log messages.
- Exceptions: use custom types from `src/utils/exceptions.py`, never raw `HTTPException` in services/ml layers.
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

# Seed dev data
python scripts/seed_db.py

# Start dev server
make dev

# Run tests
make test

# Lint
make lint
```

## Testing

- Tests live in `tests/unit/`, `tests/integration/`, `tests/e2e/`.
- Test fixtures (images, embeddings) go in `tests/fixtures/`.
- Shared fixtures are in `tests/conftest.py`.
- Use `pytest-asyncio` for async tests. Config: `asyncio_mode = "auto"` in pyproject.toml.

## Dependencies

- **Web**: FastAPI, Uvicorn, Pydantic v2
- **Image**: OpenCV (headless), NumPy, Pillow
- **ML**: InsightFace (RetinaFace + ArcFace), ONNX Runtime, PaddleOCR, Ultralytics YOLOv8
- **DB**: PostgreSQL 16 + pgvector, SQLAlchemy 2 (async), Alembic, asyncpg
- **Queue**: Celery + Redis
- **Auth**: API keys (SHA-256 hashed), python-jose (JWT for future)
- **Observability**: structlog, prometheus-client

Do not add new dependencies without justification. Prefer existing libraries over new ones.

## Things to Avoid

- Never load ML models inside request handlers. Use the registry.
- Never store uploaded images to disk or database. Process in memory, discard after.
- Never log image data or embeddings. Log only request IDs, endpoints, and timings.
- Never commit `.env` files or API keys.
- Never import across layers incorrectly (e.g., `api/` importing from `db/` directly).
- Never use synchronous database calls in FastAPI request handlers. Always use async SQLAlchemy. (Exception: Celery workers **must** use sync sessions via `src/db/sync_session.py` because asyncpg cannot run inside Celery.)
- Never hardcode thresholds, URLs, or secrets. Use `src/config.py`.

## ML Features

### Blur Detection

Two systems coexist:

1. **Laplacian-based detector** (`src/ml/blur/detector.py`): Fast coarse blur gate using Laplacian variance and FFT spectral analysis. Suitable for obvious blur; cannot distinguish blur types or detect spatially-varying blur. Always available.
2. **YOLOv8n-cls classifier** (`src/ml/blur/classifier.py`): 4-class CNN classifier (sharp, defocused_object_portrait, defocused_blurred, motion_blurred). Supports GPU via `USE_GPU` config. Targeted detection enforces a minimum confidence floor (`BLUR_DETECTION_MIN_CONFIDENCE`, default 0.5). Requires trained ONNX model at `models/blur_classifier/blur_classifier.onnx`. Optional — loads only if model file exists.

API endpoints:
- `POST /api/v1/blur/detect` — Laplacian-based (always available)
- `POST /api/v1/blur/classify` — CNN classifier (optional `blur_type` param for targeted detection)
- `POST /api/v1/blur/detect/batch` — Batch Laplacian detection via Celery
- `POST /api/v1/blur/classify/batch` — Batch classification via Celery

### Face Recognition

Pipeline: InsightFace (RetinaFace detection + ArcFace embedding) → pgvector cosine similarity search.

- `POST /api/v1/faces/detect` — Detect faces, return count and bounding boxes
- `POST /api/v1/faces/enroll` — Detect face, store embedding in DB with person name. Faces below `FACE_MIN_ENROLLMENT_CONFIDENCE` (default 0.7) are skipped; returns `LOW_QUALITY` error if all faces are below threshold.
- `POST /api/v1/faces/search` — Detect face, search DB for matches (cosine similarity)
- `POST /api/v1/faces/compare` — Compare two face images directly
- `DELETE /api/v1/faces/persons/{id}` — Remove person and their embeddings
- `POST /api/v1/faces/search/batch` — Batch face operations via Celery

### Bib Number Recognition

Pipeline: PaddleOCR (PP-OCRv5) on full image. Future: YOLO bib detection → crop → OCR. Bib text is cleaned via regex `[A-Za-z0-9\-_]` to support alphanumeric bibs with hyphens and underscores. Minimum character count configurable via `BIB_MIN_CHARS` (default 2).

- `POST /api/v1/bibs/recognize` — Detect and read bib numbers
- `POST /api/v1/bibs/recognize/batch` — Batch recognition via Celery

## Async Batch Processing

All batch endpoints use the same pattern:

1. Accept multiple files via multipart upload (max 100)
2. Create a job record in the DB
3. Base64-encode files and queue a Celery task
4. Return HTTP 202 with `job_id` and `poll_url`
5. Poll `GET /api/v1/jobs/{job_id}` for status and results
6. Webhook dispatch on completion/failure

Key files:
- `src/workers/celery_app.py` — Celery configuration (Redis broker/backend)
- `src/workers/model_loader.py` — Loads ML models once per worker process via `worker_process_init` signal
- `src/workers/helpers.py` — Shared task utils (base64 decode, job progress, webhook dispatch)
- `src/workers/tasks/` — Task implementations (blur, bib, face)
- `src/db/sync_session.py` — Sync SQLAlchemy sessions for Celery (can't use asyncpg in workers)
- `src/db/repositories/sync_*.py` — Sync repositories for Celery tasks

## Blur Classifier Training Pipeline

Training the YOLOv8n-cls blur classifier follows this workflow:

```bash
# 1. Prepare dataset (augment + split into train/val)
python scripts/prepare_blur_dataset.py

# 2. Train the model (early stopping handles convergence)
python scripts/train_blur_classifier.py

# 3. Export to ONNX for production inference
python scripts/export_blur_classifier.py

# 4. Run tests
pytest tests/test_blur_classifier.py -v
```

Training images live in `Training-Images/` (gitignored). Model artifacts live in `models/blur_classifier/` (gitignored except `manifest.json`). See `docs/phase-plan-for-blur-detection-training.md` for the full training plan, dataset details, blur detection logic rules, and accuracy targets.

## Face+Bib Detector Training Pipeline

Training the combined YOLOv8n face+bib detector follows this workflow:

```bash
# 1. Auto-annotate images using InsightFace + PaddleOCR
python scripts/auto_annotate_face_bib.py

# 2. Train the combined face+bib detector
python scripts/train_face_bib_detector.py

# 3. Export to ONNX for production inference
python scripts/export_face_bib_detector.py
```

Training images and annotations live in `Training-Images/face_bib_detection/`. See `docs/phase-plan-face-bibnumber-training.md` for the full training plan, dataset details, and accuracy targets.

## Infrastructure Requirements

| Service | Purpose | Required for |
|---------|---------|-------------|
| PostgreSQL 16 + pgvector | DB for jobs, webhooks, face embeddings | All features |
| Redis | Celery broker/backend, rate limiting | Batch processing, rate limiting |
| Celery worker | Async task execution | Batch endpoints only |

Start with: `docker compose up db redis -d`
