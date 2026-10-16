# Tech Stack

Every library, framework, and tool used in QuickPitik, with explanations of why each was chosen.

---

## Web Framework

### FastAPI
- **What it is**: Modern Python web framework for building APIs
- **Why chosen**:
  - Async support (non-blocking I/O for database, Redis, webhook calls)
  - Auto-generates Swagger documentation from code (open `/docs` in browser)
  - Built-in request validation via Pydantic
  - Dependency injection system (great for sharing model registry across endpoints)
- **Alternative considered**: Flask. Rejected because Flask is synchronous by default and lacks built-in validation/docs.

### Uvicorn
- **What it is**: ASGI server that runs FastAPI
- **Why chosen**: Fastest Python ASGI server. Supports multiple workers for production.

### Pydantic v2
- **What it is**: Data validation library
- **Why chosen**: Defines request/response schemas as Python classes. Validates input automatically. v2 is 5-50x faster than v1.

---

## Image Processing

### OpenCV (opencv-python-headless)
- **What it is**: Computer vision library (originally from Intel)
- **Why chosen**: Industry standard. Used for image decoding, color conversion, and the Laplacian filter in blur detection.
- **Why "headless"**: The headless variant doesn't include GUI dependencies (we don't need windows/displays on a server).

### NumPy
- **What it is**: Numerical computing library
- **Why chosen**: All images are represented as NumPy arrays. Also used for FFT in blur detection.

### Pillow (PIL)
- **What it is**: Image processing library
- **Why chosen**: Used for image format verification (magic byte checking) and EXIF handling. Lighter than OpenCV for these specific tasks.

---

## ML Models

### Blur Detection: classical CV + two-stage blur classifier

Two models cooperate:

- **Laplacian/FFT detector (`src/ml/blur/detector.py`)** — no external model, just OpenCV + NumPy. Computes Laplacian variance (edge energy; blurry images have fewer edges → lower variance) and an optional FFT high-frequency ratio. All classical routes decode to `BLUR_DETECT_DECODE_DIM` and normalize the score to a 640-pixel reference; scores still depend on image preparation.
- **Two-stage blur classifier (`src/ml/blur/classifier.py`)** — whole-frame rule, then person detector + per-person sharpness + logistic rule; four classes reported (`sharp`, `defocused_object_portrait`, `defocused_blurred`, `motion_blurred`). Current quality evidence is in `docs/ai-system-overview.md`. Loaded only if the person detector and the rule file are present; used by `POST /blur/classify` and its streaming/batch/mega variants.

Face recognition (InsightFace) and bib recognition (YOLOv8 + PaddleOCR) were removed on 2026-10-16; the backend uses AWS Rekognition for both.

Nothing in the served code uses the optional C++ extension. Its blur functions remain exported for benchmarks, but the served classical detector uses one OpenCV/NumPy score contract in every build.

### ONNX Runtime
- **What it is**: High-performance inference engine from Microsoft
- **Why chosen**: Runs models exported to ONNX format. Faster than PyTorch for inference. Automatically uses GPU (CUDA) if available, falls back to CPU if not. Means we don't need the full PyTorch installation at runtime.

---

## Database

### PostgreSQL 16
- **What it is**: Relational database
- **Why chosen**: Rock-solid, supports complex queries. Holds jobs, webhook subscriptions and API keys. The retired `persons` / `face_embeddings` tables are still there.

### pgvector
- **What it is**: PostgreSQL extension for vector similarity search
- **Why it is still installed**: the retired `face_embeddings` table and its migrations use pgvector's vector column type. No served code runs a vector search today.

### SQLAlchemy 2.x (async)
- **What it is**: Python ORM (Object-Relational Mapper)
- **Why chosen**: Maps database tables to Python classes. Async support via asyncpg driver. Industry standard.

### Alembic
- **What it is**: Database migration tool
- **Why chosen**: Generates migration scripts when you change the database schema. Tracks which migrations have been applied. Works with SQLAlchemy models.

---

## Task Queue & Cache

### Celery
- **What it is**: Distributed task queue
- **Why chosen**: When batch processing (100+ images), the API queues tasks to Celery workers instead of blocking. Workers can run on separate machines with GPUs. Supports task chaining and parallel fan-out.

### Redis
- **What it is**: In-memory key-value store
- **Why chosen**: Serves three roles:
  1. **Celery broker**: Message queue for background tasks
  2. **Rate limiting**: Atomic token bucket counters per API key
  3. **Cache**: Cached API key lookups (5-minute TTL) and blur detection results (keyed by image SHA-256 hash)

---

## Authentication & Security

### API Key hashing (SHA-256)
- Keys are never stored in plain text. Only the SHA-256 hash is in the database.
- Checked against Redis cache first (fast), database fallback.

### PyJWT + cryptography (JWT)
- **What it is**: JSON Web Token library. Replaced `python-jose` (which had an unpatched CVE, 2024-33664).
- **Why included**: Future support for JWT-based auth from mobile/web apps. The backend would issue JWTs and the AI API would validate them using `JWT_PUBLIC_KEY` (RS256). The infrastructure is in place but no endpoint currently validates JWTs — only API keys are enforced today.

### bcrypt
- **What it is**: Password-hashing library.
- **Why included**: Available for any future password hashing need; not used by API-key auth (which is SHA-256 hashed).

### cryptography (Fernet)
- **Why included**: Encrypts webhook secrets at rest when `WEBHOOK_SECRET_KEY` is set. Plaintext is used as a fallback with a startup warning.

---

## Observability

### structlog
- **What it is**: Structured logging library
- **Why chosen**: Outputs JSON-formatted log lines (not plain text). Every log entry includes request_id, timestamp, and context. Machine-parseable for log aggregation tools (ELK, Datadog, CloudWatch).

### prometheus-client + prometheus-fastapi-instrumentator
- **What it is**: Metrics collection for monitoring.
- **Why chosen**: Provides request counts, latency histograms, and in-progress gauges. Standard format consumed by Prometheus + Grafana dashboards.
- **Current status**: `/metrics` is mounted in `src/main.py`. In `DEBUG=true` it is open (for local scraping); in production the endpoint requires a valid `X-API-Key` and is excluded from Swagger schema.

---

## Development Tools

### Ruff
- **What it is**: Python linter and formatter (replaces flake8, isort, black)
- **Why chosen**: Extremely fast (written in Rust). Single tool for both linting and formatting.

### mypy
- **What it is**: Static type checker for Python
- **Why chosen**: Catches type errors before runtime. Configured in strict mode.

### pytest
- **What it is**: Testing framework
- **Why chosen**: Standard Python testing. With pytest-asyncio for async test support and pytest-cov for coverage reports.

---

## C++ Acceleration

### pybind11
- **What it is**: Library for creating Python bindings for C++ code.
- **Why chosen**: First-class NumPy array support (zero-copy). C++ stays pure, binding layer is thin. More maintainable than Cython or ctypes.

### CMake + scikit-build-core + Ninja
- **What it is**: Build system for compiling the extension.
- **Why chosen**: CMake is the standard C++ build system. scikit-build-core integrates CMake with Python's `pip install` workflow; Ninja is the fast generator. `pip install -e ".[cpp]"` compiles the extension; `python build_cpp.py` runs a direct CMake build (handy on Windows when `pip` misdetects MSVC).

The compiled artifact (`_quickpitik_cpp.cp<ver>-<platform>.pyd` on Windows, `.so` on Linux) lives at the project root so Python can `import _quickpitik_cpp` without path tricks. See `docs/cpp-integration.md` for benchmarks and the fallback pattern.
