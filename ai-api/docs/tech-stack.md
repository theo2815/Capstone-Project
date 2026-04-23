# Tech Stack

Every library, framework, and tool used in EventAI, with explanations of why each was chosen.

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
- **Why chosen**: All images are represented as NumPy arrays. Also used for FFT in blur detection and cosine similarity in face matching.

### Pillow (PIL)
- **What it is**: Image processing library
- **Why chosen**: Used for image format verification (magic byte checking) and EXIF handling. Lighter than OpenCV for these specific tasks.

---

## ML Models

### Blur Detection: classical CV + YOLOv8n-cls classifier

Two models cooperate:

- **Laplacian/FFT detector (`src/ml/blur/detector.py`)** — no external model, just OpenCV + NumPy. Computes Laplacian variance (edge energy; blurry images have fewer edges → lower variance) and an optional FFT high-frequency ratio. Normalized to a 640-pixel linear reference so results are resolution-independent. Fast enough to call on every upload.
- **YOLOv8n-cls classifier (`src/ml/blur/classifier.py`)** — custom-trained 4-class ONNX model (`sharp`, `defocused_object_portrait`, `defocused_blurred`, `motion_blurred`). Achieved 98.68% top-1 accuracy (100% on the sharp class). Loaded only if the ONNX file is present; used by `POST /blur/classify` and its streaming/batch/mega variants.

Both paths accept an optional C++ fast path (`laplacian_variance`, `fft_hf_ratio`, `classify_preprocess`) from the `_eventai_cpp` extension, falling back to pure NumPy/OpenCV when the extension isn't compiled.

### Face Recognition: InsightFace (RetinaFace + ArcFace)

#### RetinaFace (face detection)
- **What it does**: Finds where faces are in an image (bounding boxes + 5 landmarks per face)
- **Why chosen**: Handles varied angles, occlusions, and small faces better than alternatives (MTCNN, Haar cascades)
- **Performance**: ~80ms CPU, ~8ms GPU

#### ArcFace (face embedding)
- **What it does**: Converts a detected face into a 512-number vector (embedding). Two photos of the same person produce vectors that are close together (high cosine similarity).
- **Why chosen**: State-of-the-art accuracy. Pre-trained model available in ONNX format.
- **How matching works**: Compute cosine similarity between two embeddings. If similarity > 0.4, it's likely the same person.

#### InsightFace library
- **What it is**: Python package that bundles RetinaFace and ArcFace together with a simple API
- **Why chosen**: Unified interface for both detection and embedding. Uses ONNX Runtime backend (fast, no PyTorch runtime dependency).

### Bib Detection: YOLOv8-nano (Ultralytics)
- **What it does**: Object detection model that finds bib number regions in photos.
- **Current status**: Custom-trained ONNX at `models/bib_detection/yolov8n_bib.onnx` is in place. `BibDetector` loads it via Ultralytics; for safety it refuses non-`.onnx` paths (no pickle).
- **Why YOLOv8**: Good accuracy/speed trade-off for small objects. Nano variant keeps memory low.

### Bib OCR: PaddleOCR 3.x (PP-OCRv5)
- **What it does**: Reads text from cropped bib regions.
- **Why chosen over alternatives**:
  - **vs Tesseract**: Tesseract needs heavy preprocessing and performs poorly on photographed text (designed for scanned documents).
  - **vs EasyOCR**: Slower inference, less accurate on numeric-only text.
  - PaddleOCR is strong on text "in the wild" (real-world photos) — which is exactly what bib numbers are.
- **Post-processing**: Results are filtered with a strict character regex (`[A-Za-z0-9\-_]`) and a substitution table for common OCR confusions (`O→0`, `I→1`, `S→5`, `B→8`, `Z→2`). A `BIB_MIN_CHARS` threshold (default 2) drops candidates with too few digits.
- **Windows compatibility**: `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` is set by default because PaddlePaddle 3.x's PIR + oneDNN integration crashes on some Windows setups.

### ONNX Runtime
- **What it is**: High-performance inference engine from Microsoft
- **Why chosen**: Runs models exported to ONNX format. Faster than PyTorch for inference. Automatically uses GPU (CUDA) if available, falls back to CPU if not. Means we don't need the full PyTorch installation at runtime.

---

## Database

### PostgreSQL 16
- **What it is**: Relational database
- **Why chosen**: Rock-solid, supports complex queries, has the pgvector extension for vector search. One database for both relational data (persons, jobs, API keys) and vector data (face embeddings).

### pgvector
- **What it is**: PostgreSQL extension for vector similarity search
- **Why chosen**: Stores 512-dimensional face embeddings as a native column type. Supports HNSW index for fast approximate nearest neighbor search (sub-10ms for up to 1M embeddings). Keeps embeddings co-located with metadata (no separate vector database needed at startup scale).
- **Alternative considered**: Milvus, Pinecone (dedicated vector databases). Overkill for <1M embeddings. pgvector avoids the complexity of a separate service.

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

The compiled artifact (`_eventai_cpp.cp<ver>-<platform>.pyd` on Windows, `.so` on Linux) lives at the project root so Python can `import _eventai_cpp` without path tricks. See `docs/cpp-integration.md` for benchmarks and the fallback pattern.
