# Folder Structure

Every file and folder inside `ai-api/` explained.

```
ai-api/
│
├── pyproject.toml              # Project manifest. Lists all dependencies,
│                               # Python version, build tools, linter config.
│                               # This is the "package.json" of Python.
│
├── Makefile                    # Developer shortcuts.
│                               # "make dev" instead of long uvicorn commands.
│
├── Dockerfile                  # Production Docker image (multi-stage, optimized).
├── Dockerfile.dev              # Development Docker image (with hot-reload).
├── docker-compose.yml          # Defines 4 services: api, celery, postgres, redis.
├── docker-compose.gpu.yml      # GPU override for production deployment.
├── .dockerignore               # Files Docker should ignore when building.
│
├── alembic.ini                 # Config for Alembic (database migration tool).
├── .env.example                # Template for environment variables.
├── .gitignore                  # Files git should ignore.
│
├── CMakeLists.txt              # CMake config for C++ extension (C++17, pybind11, AVX2).
├── build_cpp.py                # Build script for C++ extension. Auto-detects MSVC
│                               # vcvarsall on Windows. Usage: python build_cpp.py
│
│
├── src/                        # === ALL APPLICATION CODE LIVES HERE ===
│   │
│   ├── __init__.py             # Makes src/ a Python package.
│   ├── main.py                 # APPLICATION ENTRY POINT.
│   │                           # Creates the FastAPI app. Defines startup
│   │                           # (load models, connect DB) and shutdown
│   │                           # (unload models, close connections).
│   │
│   ├── config.py               # CONFIGURATION.
│   │                           # Reads environment variables into a typed
│   │                           # Settings object. Every configurable value
│   │                           # (thresholds, URLs, limits) is defined here.
│   │
│   ├── dependencies.py         # DEPENDENCY INJECTION.
│   │                           # Functions that FastAPI calls to provide
│   │                           # shared objects (model registry, settings,
│   │                           # Redis) to route handlers.
│   │
│   │
│   ├── api/                    # === LAYER 1: HTTP ENDPOINTS ===
│   │   │                       # Thin controllers. Receive requests, call
│   │   │                       # services, return responses. No logic here.
│   │   │
│   │   ├── v1/                 # Version 1 of the API
│   │   │   ├── router.py       # Combines all v1 routers into one.
│   │   │   ├── health.py       # GET /health and GET /health/ready
│   │   │   ├── blur.py         # POST /blur/detect, POST /blur/detect/batch,
│   │   │   │                   #   POST /blur/classify, POST /blur/classify/batch
│   │   │   ├── faces.py        # POST /faces/detect, /enroll, /search, /compare,
│   │   │   │                   #   /search/batch
│   │   │   │                   # GET /faces/persons/{id}
│   │   │   │                   # DELETE /faces/persons/{id}
│   │   │   ├── bibs.py         # POST /bibs/recognize, POST /bibs/recognize/batch
│   │   │   ├── jobs.py         # GET /jobs/{id}
│   │   │   └── webhooks.py     # POST/GET/DELETE /webhooks
│   │   │
│   │   └── v2/                 # Future API version (empty placeholder).
│   │                           # When breaking changes are needed, v2
│   │                           # routes go here while v1 stays working.
│   │
│   │
│   ├── schemas/                # === PYDANTIC MODELS ===
│   │   │                       # Define the exact shape of every request
│   │   │                       # and response. Used for validation, docs,
│   │   │                       # and type safety.
│   │   │
│   │   ├── common.py           # APIResponse envelope (wraps all responses),
│   │   │                       # HealthResponse, JobStatus enum.
│   │   ├── blur.py             # BlurDetectionResponse, BlurMetrics,
│   │   │                       # BlurClassificationResponse, BlurClassProbabilities,
│   │   │                       # BlurTypeDetectionResponse, BlurType enum.
│   │   ├── faces.py            # BoundingBox, FaceDetection, FaceSearchResult,
│   │   │                       # FaceEnrollResponse, FaceCompareResponse, etc.
│   │   ├── bibs.py             # BibDetection, BibRecognitionResponse.
│   │   ├── jobs.py             # JobCreateResponse, JobStatusResponse.
│   │   └── webhooks.py         # WebhookCreateRequest, WebhookResponse.
│   │
│   │
│   ├── services/               # === LAYER 2: BUSINESS LOGIC ===
│   │   │                       # Contains the rules. Orchestrates ML models
│   │   │                       # and database operations.
│   │   │
│   │   ├── image_service.py    # Validates uploaded images: file type, size,
│   │   │                       # dimensions, magic bytes, EXIF stripping.
│   │   ├── blur_service.py     # Runs blur detection with optional threshold.
│   │   ├── face_service.py     # Face detection and embedding extraction.
│   │   ├── bib_service.py      # Bib detection + OCR pipeline.
│   │   ├── job_service.py      # Creates and tracks async batch jobs.
│   │   └── webhook_service.py  # Dispatches webhook callbacks to subscribers.
│   │
│   │
│   ├── ml/                     # === LAYER 3: ML MODEL WRAPPERS ===
│   │   │                       # Each file wraps one AI library. These know
│   │   │                       # how to run inference but nothing about HTTP
│   │   │                       # or databases.
│   │   │
│   │   ├── registry.py         # MODEL REGISTRY (most important file).
│   │   │                       # Loads all models once at startup.
│   │   │                       # Stores them in memory. All requests share
│   │   │                       # the same model instances.
│   │   │
│   │   ├── blur/
│   │   │   ├── detector.py     # BlurDetector class.
│   │   │   │                   # Method 1: Laplacian variance (cv2.Laplacian).
│   │   │   │                   # Method 2: FFT spectral analysis (np.fft).
│   │   │   │                   # Has C++ fast path with Python fallback.
│   │   │   │                   # Returns: is_blurry + confidence + metrics.
│   │   │   └── classifier.py   # BlurClassifier. ONNX-based blur/sharp
│   │   │                       # classification model (YOLOv8n-cls trained
│   │   │                       # on custom blur dataset, 98.68% accuracy).
│   │   │
│   │   ├── faces/
│   │   │   ├── detector.py     # FaceDetector. Uses RetinaFace via InsightFace.
│   │   │   │                   # Returns bounding boxes + landmarks.
│   │   │   ├── embedder.py     # FaceEmbedder. Uses RetinaFace + ArcFace.
│   │   │   │                   # Returns 512-dim embedding vectors per face.
│   │   │   └── matcher.py      # Cosine similarity matching. Compares a query
│   │   │                       # embedding against a database of embeddings.
│   │   │                       # Has C++ fast path with NumPy fallback.
│   │   │
│   │   └── bibs/
│   │       ├── detector.py     # BibDetector. Uses YOLOv8 to find bib regions.
│   │       └── recognizer.py   # BibRecognizer. Uses PaddleOCR to read numbers
│   │                           # from cropped bib images.
│   │
│   │
│   ├── cpp/                    # === C++ EXTENSION SOURCE (Phase 6) ===
│   │   │                       # pybind11 C++ module for performance-critical
│   │   │                       # paths. Compiled as _eventai_cpp.pyd/.so.
│   │   │                       # All functions release the GIL for concurrency.
│   │   │
│   │   ├── bindings.cpp        # PYBIND11_MODULE — exposes all C++ functions
│   │   │                       # and structs to Python.
│   │   ├── face_ops.h/.cpp     # cosine_similarity, batch_cosine_topk
│   │   │                       # (partial_sort, AVX2 vectorization).
│   │   ├── blur_ops.h/.cpp     # laplacian_variance (single-pass),
│   │   │                       # fft_hf_ratio (radix-2 Cooley-Tukey 2D FFT),
│   │   │                       # batch_blur_metrics.
│   │   └── preprocess_ops.h/.cpp  # bgr_to_gray, resize_gray (bilinear).
│   │
│   │
│   ├── db/                     # === LAYER 4: DATABASE ===
│   │   │
│   │   ├── session.py          # Creates the async database connection pool.
│   │   │                       # Provides get_session() for database access.
│   │   │                       # Handles init, close, and health check.
│   │   │
│   │   ├── sync_session.py     # Sync SQLAlchemy engine/session for Celery
│   │   │                       # workers (can't use asyncpg in Celery).
│   │   │
│   │   ├── models.py           # DATABASE TABLES defined as Python classes:
│   │   │                       # - Person (name, metadata)
│   │   │                       # - FaceEmbedding (512-dim vector via pgvector)
│   │   │                       # - Job (status, progress, results)
│   │   │                       # - WebhookSubscription (URL, events)
│   │   │                       # - APIKey (hashed key, scopes, rate tier)
│   │   │
│   │   ├── repositories/       # CRUD operations for each table:
│   │   │   ├── face_repo.py    #   Async — create/delete persons, store
│   │   │   │                   #   embeddings, vector similarity search.
│   │   │   ├── job_repo.py     #   Async — create/update/complete/fail jobs.
│   │   │   ├── webhook_repo.py #   Async — manage webhook subscriptions.
│   │   │   ├── sync_face_repo.py   # Sync — pgvector search for Celery tasks.
│   │   │   ├── sync_job_repo.py    # Sync — job progress/complete/fail for Celery.
│   │   │   └── sync_webhook_repo.py # Sync — webhook lookup for Celery.
│   │   │
│   │   └── migrations/
│   │       ├── env.py          # Alembic migration runner (async-aware).
│   │       └── versions/       # Generated migration scripts go here.
│   │
│   │
│   ├── middleware/              # === CROSS-CUTTING CONCERNS ===
│   │   │                       # These run on EVERY request before the
│   │   │                       # route handler executes.
│   │   │
│   │   ├── auth.py             # API key authentication.
│   │   │                       # Extracts X-API-Key header, hashes it,
│   │   │                       # looks up in Redis cache or database.
│   │   │                       # Returns key metadata (scopes, rate tier).
│   │   │
│   │   ├── rate_limit.py       # Token bucket rate limiting via Redis.
│   │   │                       # Free: 60/min, Pro: 300/min, Internal: 1000/min.
│   │   │                       # Returns 429 with Retry-After header.
│   │   │
│   │   ├── request_id.py       # Assigns a unique UUID to every request.
│   │   │                       # Included in logs and responses for tracing.
│   │   │
│   │   └── cors.py             # CORS configuration.
│   │                           # Controls which frontend domains can call the API.
│   │
│   │
│   ├── workers/                # === BACKGROUND TASK PROCESSING ===
│   │   │                       # For batch operations that take too long
│   │   │                       # for a synchronous HTTP response.
│   │   │
│   │   ├── celery_app.py       # Celery configuration. Connects to Redis
│   │   │                       # as the message broker.
│   │   ├── model_loader.py     # Loads ML models in Celery worker processes
│   │   │                       # via worker_process_init signal (once per process).
│   │   ├── helpers.py          # Shared task utilities: base64 image decode,
│   │   │                       # job progress updates, webhook dispatch.
│   │   │
│   │   └── tasks/
│   │       ├── blur_tasks.py   # Batch blur detection task.
│   │       ├── face_tasks.py   # Batch face processing (detect + search).
│   │       ├── bib_tasks.py    # Batch bib recognition task.
│   │       └── webhook_tasks.py # Delivers webhook callbacks to registered
│   │                           # URLs with HMAC signature and retries.
│   │
│   │
│   └── utils/                  # === SHARED UTILITIES ===
│       ├── exceptions.py       # Custom error types: ImageValidationError,
│       │                       # ModelNotLoadedError, AuthenticationError, etc.
│       │                       # Each has an HTTP status code.
│       │
│       ├── image_utils.py      # Standalone image validation function.
│       │                       # Checks type, size, dimensions, magic bytes.
│       │
│       ├── logging.py          # Structured JSON logging via structlog.
│       │                       # Every log line has request_id, timestamp.
│       │
│       └── metrics.py          # Prometheus metrics: request counts,
│                               # latency histograms, inference times.
│
│
├── tests/                      # === TESTS ===
│   ├── conftest.py             # Shared test fixtures (test client, etc.)
│   ├── test_blur_detector.py   # Unit tests for BlurDetector.
│   ├── test_blur_classifier.py # Unit tests for BlurClassifier.
│   ├── test_blur_endpoint.py   # Integration tests for POST /blur/detect.
│   ├── test_face_matcher.py    # Unit tests for face matcher.
│   ├── test_face_endpoints.py  # Integration tests for face endpoints.
│   ├── test_bib_recognizer.py  # Unit tests for BibRecognizer.
│   ├── test_bib_endpoint.py    # Integration tests for POST /bibs/recognize.
│   ├── test_batch_endpoints.py # Integration tests for batch processing.
│   ├── test_cpp_extension.py   # Tests for C++ extension (numerical parity).
│   ├── unit/                   # Additional unit tests.
│   ├── integration/            # Additional integration tests.
│   ├── e2e/                    # End-to-end tests with real models.
│   └── fixtures/
│       ├── images/             # Test images (blurry, sharp, faces, bibs).
│       └── embeddings/         # Pre-computed test embeddings.
│
│
├── benchmarks/                 # === PERFORMANCE BENCHMARKS ===
│   └── bench_cpp_vs_python.py  # Timing comparisons for C++ vs Python paths.
│
│
├── models/                     # === ML MODEL FILES ===
│   │                           # This folder is GIT-IGNORED (models are huge).
│   │                           # Models are downloaded at build/start time.
│   ├── manifest.json           # Lists all required models with sources
│   │                           # and notes on how to obtain them.
│   └── blur_classifier/        # Trained blur classifier (Phase 5):
│       ├── blur_classifier.onnx #   ONNX model — 98.68% accuracy.
│       └── class_names.json    #   Class labels: ["sharp", "defocused_object_portrait",
│                               #     "defocused_blurred", "motion_blurred"].
│
│
├── scripts/                    # === DEVELOPER SCRIPTS ===
│   ├── download_models.py      # Checks which models are present locally.
│   ├── seed_db.py              # Creates dev API key and test data.
│   ├── benchmark.py            # Performance benchmarks (Python vs C++).
│   ├── prepare_blur_dataset.py # Prepares blur/sharp training dataset.
│   ├── train_blur_classifier.py # Trains YOLOv8n-cls blur classifier.
│   ├── export_blur_classifier.py # Exports trained blur model to ONNX.
│   ├── auto_annotate_face_bib.py # Auto-annotates images using InsightFace +
│   │                           # PaddleOCR for face+bib detection training.
│   ├── train_face_bib_detector.py # Trains YOLOv8n combined face+bib detector.
│   └── export_face_bib_detector.py # Exports trained face+bib model to ONNX
│                               # at models/bib_detection/yolov8n_bib.onnx.
│
│
└── docs/                       # === DOCUMENTATION (you are here) ===
```
