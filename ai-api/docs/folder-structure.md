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
│   │   │   ├── blur.py         # POST /blur/detect
│   │   │   ├── faces.py        # POST /faces/detect, /enroll, /search, /compare
│   │   │   │                   # GET /faces/persons/{id}
│   │   │   │                   # DELETE /faces/persons/{id}
│   │   │   ├── bibs.py         # POST /bibs/recognize
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
│   │   ├── blur.py             # BlurDetectionResponse, BlurMetrics.
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
│   │   │                       # Loads all 4 models once at startup.
│   │   │                       # Stores them in memory. All requests share
│   │   │                       # the same model instances.
│   │   │
│   │   ├── blur/
│   │   │   └── detector.py     # BlurDetector class.
│   │   │                       # Method 1: Laplacian variance (cv2.Laplacian).
│   │   │                       # Method 2: FFT spectral analysis (np.fft).
│   │   │                       # Returns: is_blurry + confidence + metrics.
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
│   ├── db/                     # === LAYER 4: DATABASE ===
│   │   │
│   │   ├── session.py          # Creates the async database connection pool.
│   │   │                       # Provides get_session() for database access.
│   │   │                       # Handles init, close, and health check.
│   │   │
│   │   ├── models.py           # DATABASE TABLES defined as Python classes:
│   │   │                       # - Person (name, metadata)
│   │   │                       # - FaceEmbedding (512-dim vector via pgvector)
│   │   │                       # - Job (status, progress, results)
│   │   │                       # - WebhookSubscription (URL, events)
│   │   │                       # - APIKey (hashed key, scopes, rate tier)
│   │   │
│   │   ├── repositories/       # CRUD operations for each table:
│   │   │   ├── face_repo.py    #   Create/delete persons, store embeddings,
│   │   │   │                   #   vector similarity search via pgvector.
│   │   │   ├── job_repo.py     #   Create/update/complete/fail jobs.
│   │   │   └── webhook_repo.py #   Manage webhook subscriptions.
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
│   │   │
│   │   └── tasks/
│   │       ├── blur_tasks.py   # Batch blur detection (stub, Phase 5).
│   │       ├── face_tasks.py   # Batch face processing (stub, Phase 5).
│   │       ├── bib_tasks.py    # Batch bib recognition (stub, Phase 5).
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
│   ├── unit/                   # Tests for individual functions/classes.
│   ├── integration/            # Tests that hit real endpoints.
│   ├── e2e/                    # End-to-end tests with real models.
│   └── fixtures/
│       ├── images/             # Test images (blurry, sharp, faces, bibs).
│       └── embeddings/         # Pre-computed test embeddings.
│
│
├── models/                     # === ML MODEL FILES ===
│   │                           # This folder is GIT-IGNORED (models are huge).
│   │                           # Models are downloaded at build/start time.
│   └── manifest.json           # Lists all required models with sources
│                               # and notes on how to obtain them.
│
│
├── scripts/                    # === DEVELOPER SCRIPTS ===
│   ├── download_models.py      # Checks which models are present locally.
│   ├── seed_db.py              # Creates dev API key and test data.
│   └── benchmark.py            # Performance benchmarks (Python vs C++).
│
│
└── docs/                       # === DOCUMENTATION (you are here) ===
```
