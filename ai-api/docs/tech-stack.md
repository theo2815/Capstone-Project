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

### Blur Detection: OpenCV Laplacian + NumPy FFT
- **No external model needed** - uses mathematical algorithms, not neural networks
- **Laplacian variance**: A convolution filter that detects edges. Blurry images have fewer edges = lower variance. Fast (~2ms per image).
- **FFT (Fast Fourier Transform)**: Converts image to frequency domain. Blurry images lack high frequencies. Better for distinguishing motion blur from defocus blur.

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
- **What it does**: Object detection model that finds bib number regions in photos
- **Why YOLOv8**: Latest YOLO version. Nano variant is fast (~5ms GPU) while maintaining good accuracy.
- **Note**: Requires custom training on bib number datasets. The model file (`yolov8n_bib.onnx`) needs to be trained before this feature works.

### Bib OCR: PaddleOCR
- **What it does**: Reads text from images (Optical Character Recognition)
- **Why chosen over alternatives**:
  - **vs Tesseract**: Tesseract needs heavy preprocessing and performs poorly on photographed text (designed for scanned documents)
  - **vs EasyOCR**: Slower inference, less accurate on numeric-only text
  - PaddleOCR is specifically strong on text "in the wild" (real-world photos), which is exactly what bib numbers are

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

### python-jose (JWT)
- **What it is**: JSON Web Token library
- **Why included**: Future support for JWT-based auth from mobile/web apps. The backend service would issue JWTs, and the AI API would validate them using a shared public key.

---

## Observability

### structlog
- **What it is**: Structured logging library
- **Why chosen**: Outputs JSON-formatted log lines (not plain text). Every log entry includes request_id, timestamp, and context. Machine-parseable for log aggregation tools (ELK, Datadog, CloudWatch).

### prometheus-client + prometheus-fastapi-instrumentator
- **What it is**: Metrics collection for monitoring
- **Why chosen**: Exposes `/metrics` endpoint with request counts, latency histograms, and custom metrics (inference time, image sizes). Standard format consumed by Prometheus + Grafana dashboards.

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

## C++ Acceleration (Phase 6)

### pybind11
- **What it is**: Library for creating Python bindings for C++ code
- **Why chosen**: First-class NumPy array support (zero-copy). C++ stays pure, binding layer is thin. More maintainable than Cython or ctypes.

### CMake + scikit-build-core
- **What it is**: Build system for compiling C++ code
- **Why chosen**: CMake is the standard C++ build system. scikit-build-core integrates CMake with Python's `pip install` workflow so the C++ extension compiles automatically during installation.
