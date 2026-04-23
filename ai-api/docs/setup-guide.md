# Setup Guide

## Prerequisites

- **Python 3.11 – 3.14** (see `pyproject.toml`: `requires-python = ">=3.11,<3.15"`)
- **Docker Desktop** (for PostgreSQL and Redis)
- **Git**
- **(Optional) C++ toolchain** — only needed if you want to build the `_eventai_cpp` extension locally: GCC 10+/Clang 12+ on Linux, MSVC (Visual Studio Build Tools) on Windows, Xcode CLT on macOS. The app runs fine with the NumPy fallback if the extension isn't built.

## Quick Start

### 1. Clone and enter the project
```bash
cd "c:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project\ai-api"
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
# Install the project with dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

### 4. Set up environment variables
```bash
# Copy the template
copy .env.example .env

# Edit .env with your settings (the defaults work for local Docker setup)
```

### 5. Start PostgreSQL and Redis
```bash
docker compose up db redis -d
```
This starts:
- PostgreSQL (with pgvector) on port 5432
- Redis on port 6379

### 6. Run database migrations
```bash
# Migrations already exist under src/db/migrations/versions — just apply them:
alembic upgrade head
```

(Only run `alembic revision --autogenerate -m "..."` when you're adding a *new* migration after changing `src/db/models.py`.)

### 7. Seed the database (optional)
```bash
python scripts/seed_db.py
```
This inserts a development API key into the `api_keys` table. Check `scripts/seed_db.py` for the exact raw key string it prints — keep it handy for `curl` tests.

### 8. Start the API server
```bash
# Development mode (auto-reloads on code changes)
make dev

# Or manually:
uvicorn src.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### 9. Verify it works
Open your browser to: http://localhost:8000/docs

This shows the Swagger UI with all endpoints. You can test them interactively.

> **Note:** Swagger UI and `/redoc` are only available when `DEBUG=true`. If `DEBUG=false`, both return 404. The server also refuses to start with `DEBUG=true` when `ENVIRONMENT=production`.

Or via curl:
```bash
# Liveness (no auth)
curl http://localhost:8000/api/v1/health

# Readiness (requires API key — prevents leaking infra status)
curl -H "X-API-Key: <your-key>" http://localhost:8000/api/v1/health/ready

# Blur detection
curl -X POST http://localhost:8000/api/v1/blur/detect \
  -H "X-API-Key: <your-key>" \
  -F "file=@some_photo.jpg"
```

---

## Running Everything with Docker

If you don't want to install Python locally, run the entire stack in Docker:

```bash
# Build and start all services
docker compose up --build

# Or in detached mode
docker compose up --build -d

# View logs
docker compose logs -f ai-api

# Stop everything
docker compose down
```

This starts 4 containers: ai-api, celery-worker, PostgreSQL, Redis.

---

## Common Commands

| Command | What it does |
|---|---|
| `make dev` | Start API server with auto-reload |
| `make run` | Start API server in production mode |
| `make test` | Run all tests |
| `make test-cov` | Run tests with coverage report |
| `make lint` | Check code style and types |
| `make format` | Auto-format code |
| `make migrate` | Apply database migrations |
| `make migration msg="add new table"` | Create a new migration |
| `make docker-up` | Start all Docker services |
| `make docker-down` | Stop all Docker services |
| `make docker-gpu` | Start Docker services with GPU support |
| `make celery` | Start Celery worker locally |
| `make install` | Install dev dependencies |
| `make clean` | Remove cached files |

---

## Environment Variables

All configuration is done via environment variables. See `.env.example` for the full list.

**Key variables:**

| Variable | Default | Description |
|---|---|---|
| `DEBUG` | `false` | Enable debug mode (opens Swagger, relaxes auth in dev) |
| `ENVIRONMENT` | `development` | Set to `production` in prod. Startup rejects `DEBUG=true` + `production`. |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/eventai` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `MODEL_DIR` | `./models` | Root directory the registry looks in for model files |
| `USE_GPU` | `false` | Enable GPU inference (needs `[gpu]` extras + CUDA runtime) |
| `BLUR_THRESHOLD` | `100.0` | Default Laplacian variance threshold |
| `BLUR_DETECTION_MIN_CONFIDENCE` | `0.5` | Minimum confidence for targeted blur type detection |
| `FACE_SIMILARITY_THRESHOLD` | `0.4` | Default cosine similarity threshold for face match |
| `FACE_MIN_ENROLLMENT_CONFIDENCE` | `0.7` | Skip faces below this detection confidence on enroll |
| `FACE_DET_SIZE` | `640` | RetinaFace input size |
| `BIB_MIN_CHARS` | `2` | Minimum digit count for a bib candidate |
| `MAX_FILE_SIZE` | `10485760` | Max single upload in bytes (10MB) |
| `MAX_BATCH_SIZE` | `50` | Max files per `/batch` endpoint |
| `MEGA_BATCH_MAX_SIZE` | `500` | Max files per `/mega` endpoint |
| `STREAM_BATCH_MAX_SIZE` | `500` | Max files per `/detect/stream` |
| `STREAM_CLASSIFY_MAX_SIZE` | `500` | Max files per `/classify/stream` |
| `MAX_ACTIVE_JOBS_PER_KEY` | `10` | Backpressure cap on concurrent batch jobs per API key |
| `JOB_RETENTION_DAYS` | `7` | Auto-delete completed/failed jobs older than this |
| `MAX_INFERENCE_DIMENSION` | `640` | Downscale images before inference to this max side |
| `INFERENCE_SUB_BATCH_SIZE` | `50` | Sub-batch size inside Celery batch tasks |
| `ONNX_INTRA_OP_THREADS` | `6` | ONNX intra-op thread count |
| `BLOB_STORE_PATH` | `/tmp/eventai-blobs` | Shared volume for batch image staging |
| `ALLOWED_ORIGINS` | `["http://localhost:3000"]` | CORS allowed origins |
| `WEBHOOK_SECRET_KEY` | (empty) | Fernet key for webhook-secret encryption at rest |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Troubleshooting

### "Database not initialized" error
Make sure PostgreSQL is running: `docker compose up db -d`

### "Redis not available" warning
Not critical for development. Redis is optional. Rate limiting and caching will be disabled.

### Models fail to load
On first run, InsightFace and PaddleOCR download models automatically (~500MB total). Ensure you have internet access. The custom YOLOv8 bib detector ONNX must be placed manually at `models/bib_detection/yolov8n_bib.onnx` — the repo ships with one that's ready to use. The blur classifier ONNX is also optional and lives at `models/blur_classifier/blur_classifier.onnx`; if missing, `/blur/classify*` returns `MODEL_UNAVAILABLE` but `/blur/detect*` still works.

### Port 8000 already in use
Either stop the other process or change the port:
```bash
uvicorn src.main:create_app --factory --port 8080 --reload
```

### Import errors after install
Make sure your virtual environment is activated and you installed with `pip install -e ".[dev]"`.
