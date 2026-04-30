# Folder Structure

Every file and folder inside `ai-api/` explained. Matches the current codebase.

```
ai-api/
│
├── CLAUDE.md                   # Entry point for AI agents / new team members.
├── pyproject.toml              # Project manifest: deps, Python 3.11–3.14,
│                               # build tools (setuptools), ruff, mypy, pytest.
├── Makefile                    # Developer shortcuts (`make dev`, `make test`, etc.).
│
├── Dockerfile                  # Production image (multi-stage, optimized).
├── Dockerfile.dev              # Development image (with hot-reload).
├── docker-compose.yml          # 4 services: ai-api, celery-worker, db (pg+pgvector),
│                               # redis. Default dev config.
├── docker-compose.prod.yml     # Production overrides.
├── docker-compose.gpu.yml      # GPU overlay (NVIDIA runtime + USE_GPU=true).
├── nginx.conf                  # Sample nginx reverse-proxy config.
├── .dockerignore
│
├── alembic.ini                 # Alembic config (migration tool).
├── .env.example                # Template for environment variables.
├── .gitignore
│
├── CMakeLists.txt              # C++17 / pybind11 / AVX2 build config.
├── build_cpp.py                # Manual build script — auto-detects MSVC on Windows.
│                               # Usage: `python build_cpp.py`
├── _quickpitik_cpp.*.pyd/.so     # Compiled C++ extension (gitignored; produced by build).
├── yolov8n-cls.pt             # Pre-trained YOLOv8 weights used as the base for
│                               # blur classifier training (gitignored).
│
├── gen_api_key.py              # One-off script to generate a new API key.
├── insert_key.py               # One-off script to insert an API key into the DB.
│
│
├── src/                        # === ALL APPLICATION CODE LIVES HERE ===
│   │
│   ├── __init__.py
│   ├── main.py                 # APPLICATION ENTRY POINT.
│   │                           # Builds the FastAPI app, registers middleware,
│   │                           # wires the lifespan (DB init, Redis connect,
│   │                           # model load → shutdown).
│   │                           # Also mounts /metrics (auth-protected in prod).
│   │
│   ├── config.py               # CONFIGURATION.
│   │                           # Pydantic `Settings` class. Every configurable
│   │                           # value (thresholds, URLs, limits) is defined here.
│   │
│   │
│   ├── api/                    # === LAYER 1: HTTP ENDPOINTS ===
│   │   │                       # Thin controllers. No business logic here.
│   │   │
│   │   └── v1/                 # Version 1 of the API.
│   │       ├── router.py       # Combines all v1 routers into one.
│   │       ├── health.py       # GET /health, GET /health/ready
│   │       ├── blur.py         # Single/stream/batch/mega for detect + classify.
│   │       ├── faces.py        # detect/enroll/search/compare, persons CRUD,
│   │       │                   # search-batch, enroll-batch, search-mega.
│   │       ├── bibs.py         # recognize + batch + mega.
│   │       ├── jobs.py         # GET /jobs/{id} (with offset/limit pagination).
│   │       ├── webhooks.py     # POST/GET/DELETE /webhooks.
│   │       │                   # Basic SSRF check at registration time.
│   │       ├── batch_utils.py  # Shared helpers: validate_batch_files,
│   │       │                   # create_batch_job (backpressure), store_blobs_and_get_paths,
│   │       │                   # batch_accepted_response.
│   │       └── mega_batch.py   # dispatch_mega_batch — splits 500-image uploads
│   │                           # into MAX_BATCH_SIZE-sized sub-tasks via Celery chord.
│   │
│   │
│   ├── schemas/                # === PYDANTIC MODELS ===
│   │   ├── common.py           # APIResponse envelope, ErrorDetail,
│   │   │                       # HealthResponse, ReadinessResponse.
│   │   ├── blur.py             # BlurType enum, BlurMetrics, BlurDetectionResponse,
│   │   │                       # BlurClassProbabilities, BlurClassificationResponse,
│   │   │                       # BlurTypeDetectionResponse.
│   │   ├── faces.py            # BoundingBox, FaceDetection, FaceSearchResult,
│   │   │                       # FaceDetect/Search/Enroll/Compare responses,
│   │   │                       # PersonResponse, PersonListResponse.
│   │   ├── bibs.py             # BibCandidate, BibDetection, BibRecognitionResponse.
│   │   ├── jobs.py             # JobStatus enum, JobCreateResponse, JobStatusResponse.
│   │   └── webhooks.py         # ALLOWED_EVENTS, WebhookCreateRequest,
│   │                           # WebhookResponse, WebhookListResponse.
│   │
│   │
│   ├── services/               # === LAYER 2: BUSINESS LOGIC ===
│   │   │                       # Only blur currently has a dedicated service class.
│   │   │                       # Face and bib logic lives inline in the API handlers
│   │   │                       # because orchestration is still thin — introduce a
│   │   │                       # service class when the rules grow.
│   │   │
│   │   └── blur_service.py     # BlurService — wraps BlurDetector + optional
│   │                           # BlurClassifier. Used by tests; handlers also
│   │                           # call the ML layer directly today.
│   │
│   │
│   ├── ml/                     # === LAYER 3: ML MODEL WRAPPERS ===
│   │   │                       # Each file wraps one AI library. No HTTP or DB awareness.
│   │   │
│   │   ├── registry.py         # MODEL REGISTRY.
│   │   │                       # Loads all models in parallel via asyncio.gather
│   │   │                       # during FastAPI lifespan. Pre-imports torch,
│   │   │                       # insightface, ultralytics in the main thread to
│   │   │                       # avoid Windows DLL races. Releases GPU/ONNX resources
│   │   │                       # on shutdown.
│   │   │
│   │   ├── blur/
│   │   │   ├── detector.py     # BlurDetector (Laplacian variance + FFT high-
│   │   │   │                   # frequency ratio, 640px-normalized). `detect_fast`
│   │   │   │                   # skips FFT + BGR→gray for the hot path. Uses C++
│   │   │   │                   # `laplacian_variance` / `fft_hf_ratio` when available.
│   │   │   └── classifier.py   # BlurClassifier (YOLOv8n-cls, 4 classes via ONNX).
│   │   │                       # Optional — loads only if
│   │   │                       # models/blur_classifier/blur_classifier.onnx exists.
│   │   │                       # Uses C++ `classify_preprocess` when available.
│   │   │
│   │   ├── faces/
│   │   │   ├── embedder.py     # FaceEmbedder — wraps InsightFace `buffalo_l`
│   │   │   │                   # (RetinaFace + ArcFace). Drops unused genderage +
│   │   │   │                   # extra landmark sub-models at load (~40% less
│   │   │   │                   # compute/memory).
│   │   │   └── matcher.py      # cosine_similarity + find_matches (top-K).
│   │   │                       # Uses C++ `batch_cosine_topk` when available.
│   │   │
│   │   └── bibs/
│   │       ├── detector.py     # BibDetector — loads a custom YOLOv8n ONNX via
│   │       │                   # Ultralytics. Refuses non-.onnx paths for safety.
│   │       └── recognizer.py   # BibRecognizer — PaddleOCR 3.x (PP-OCRv5).
│   │                           # `recognize_batch` parallelises `predict()` calls
│   │                           # across an OCR_MAX_WORKERS thread pool. OCR digits
│   │                           # are cleaned with a regex filter + substitution map
│   │                           # (O→0, I→1, S→5, etc.).
│   │
│   │
│   ├── cpp/                    # === C++ EXTENSION SOURCE ===
│   │   │                       # pybind11 C++ module built as `_quickpitik_cpp`.
│   │   │                       # All functions release the GIL.
│   │   │
│   │   ├── bindings.cpp        # PYBIND11_MODULE — exposes all C++ functions.
│   │   ├── face_ops.h/.cpp     # cosine_similarity, batch_cosine_topk (partial_sort,
│   │   │                       # AVX2-friendly).
│   │   ├── blur_ops.h/.cpp     # laplacian_variance (single-pass sum+sum_sq),
│   │   │                       # fft_hf_ratio (radix-2 Cooley-Tukey 2D FFT),
│   │   │                       # batch_blur_metrics.
│   │   └── preprocess_ops.h/.cpp  # bgr_to_gray, resize_gray (bilinear),
│   │                           # classify_preprocess (fused resize + normalize +
│   │                           # transpose for YOLOv8-cls input).
│   │
│   │
│   ├── db/                     # === LAYER 4: DATABASE ===
│   │   │
│   │   ├── session.py          # Async SQLAlchemy engine (asyncpg driver).
│   │   │                       # Exposes init_db, close_db, check_db_health,
│   │   │                       # get_session_ctx (async context manager).
│   │   │
│   │   ├── sync_session.py     # Sync engine (psycopg2) for Celery workers.
│   │   │                       # asyncpg cannot run inside a Celery prefork child.
│   │   │                       # Exposes init_sync_db, close_sync_db, get_sync_session.
│   │   │
│   │   ├── models.py           # DATABASE TABLES:
│   │   │                       # - Person (id, name, api_key_id, event_id, metadata)
│   │   │                       # - FaceEmbedding (person_id, Vector(512), image_hash,
│   │   │                       #     quality_score)
│   │   │                       # - Job (job_type, status, progress, total/processed,
│   │   │                       #     result JSONB, api_key_id, timestamps)
│   │   │                       # - WebhookSubscription (url, events, secret, api_key_id)
│   │   │                       # - APIKey (key_hash, scopes, rate_tier, active)
│   │   │
│   │   ├── repositories/
│   │   │   ├── face_repo.py        # Async — persons CRUD, embeddings, batch search.
│   │   │   ├── job_repo.py         # Async — create/update/complete/fail,
│   │   │   │                       # count_active_by_key (backpressure).
│   │   │   ├── webhook_repo.py     # Async — subscription CRUD.
│   │   │   ├── sync_face_repo.py   # Sync equivalent used by Celery tasks.
│   │   │   ├── sync_job_repo.py    # Sync — plus reap_stale_jobs, cleanup_old_jobs.
│   │   │   └── sync_webhook_repo.py # Sync — list_by_event for worker dispatch.
│   │   │
│   │   └── migrations/
│   │       ├── env.py          # Alembic runner (async-aware).
│   │       └── versions/       # Migration scripts.
│   │
│   │
│   ├── middleware/             # === CROSS-CUTTING CONCERNS ===
│   │   ├── auth.py             # verify_api_key: SHA-256 hash → Redis cache → DB.
│   │   │                       # Calls check_rate_limit after auth succeeds.
│   │   │                       # check_scope helper for per-endpoint scope checks.
│   │   │                       # invalidate_api_key_cache for key revocation.
│   │   ├── rate_limit.py       # Token-bucket rate limiter (Redis Lua script).
│   │   │                       # Free=60/min, Pro=300/min, Internal=1000/min.
│   │   │                       # Stores rate_info on request.state for the header
│   │   │                       # middleware in main.py.
│   │   ├── request_id.py       # Assigns/validates X-Request-ID; puts it on
│   │   │                       # request.state.request_id.
│   │   └── cors.py             # CORS setup (allow_methods GET/POST/DELETE;
│   │                           # exposes X-Request-ID, X-RateLimit-Remaining/Reset).
│   │
│   │   (Additional middleware — TimeoutMiddleware, SecurityHeadersMiddleware,
│   │    RateLimitHeadersMiddleware — is defined inline in main.py because it
│   │    depends on runtime settings.)
│   │
│   │
│   ├── workers/                # === BACKGROUND TASK PROCESSING ===
│   │   ├── celery_app.py       # Celery config. Queues: default, blur, face, bib.
│   │   │                       # Task time limits (soft 3300s, hard 3600s),
│   │   │                       # worker_max_tasks_per_child=500,
│   │   │                       # worker_max_memory_per_child=2GB,
│   │   │                       # prefetch=4. Beat schedule: reap-stale-jobs (5 min),
│   │   │                       # cleanup-old-jobs (daily), cleanup-stale-blobs (30 min).
│   │   ├── model_loader.py     # worker_process_init signal.
│   │   │                       # Reads WORKER_QUEUES env var to decide which models
│   │   │                       # to load. Initialises sync DB engine.
│   │   ├── helpers.py          # Shared utilities:
│   │   │                       # decode_image_from_path, decode_grays_from_paths
│   │   │                       # (parallel decode via ThreadPoolExecutor),
│   │   │                       # update_job_progress (throttled), complete_job,
│   │   │                       # fail_job, dispatch_webhook_sync.
│   │   │
│   │   └── tasks/
│   │       ├── blur_tasks.py       # blur.detect_batch, blur.classify_batch.
│   │       │                       # Sub-batches of INFERENCE_SUB_BATCH_SIZE (50).
│   │       ├── face_tasks.py       # faces.process_batch (detect|search) +
│   │       │                       # faces.enroll_batch (two-phase: inference, then DB).
│   │       ├── bib_tasks.py        # bibs.recognize_batch — batch YOLO → batch OCR.
│   │       ├── webhook_tasks.py    # webhooks.deliver — SSRF DNS resolve + IP-literal
│   │       │                       # request (TOCTOU-safe). HMAC signature, retry
│   │       │                       # with exponential backoff.
│   │       └── maintenance_tasks.py # reap_stale_jobs, cleanup_old_jobs,
│   │                           # cleanup_stale_blobs, finalize_mega_batch (chord callback).
│   │
│   │
│   └── utils/                  # === SHARED UTILITIES ===
│       ├── exceptions.py       # QuickPitikError + subclasses (ImageValidationError,
│       │                       # ModelNotLoadedError, AuthenticationError,
│       │                       # RateLimitExceededError, JobNotFoundError).
│       ├── image_utils.py      # validate_and_decode (single image), validate_batch_file,
│       │                       # downscale_for_inference, get_image_dimensions.
│       │                       # Content-type allowlist, magic-byte check via PIL,
│       │                       # EXIF rotation, dimension limits (32–4096 px).
│       ├── blob_store.py       # store_batch, load_blob, cleanup_batch,
│       │                       # cleanup_stale_blobs. Atomic write (tmp→rename),
│       │                       # parallel writes for batch uploads.
│       ├── crypto.py           # Fernet-based encryption for webhook secrets
│       │                       # (WEBHOOK_SECRET_KEY). Plaintext fallback if unset.
│       ├── timeout.py          # run_with_timeout (reusable single-thread executor)
│       │                       # and run_direct. Used sparingly — Celery's soft/hard
│       │                       # time limits are the main safety net.
│       └── logging.py          # setup_logging, get_logger — structlog JSON output
│                               # with request_id context.
│
│
├── tests/                      # === TESTS ===
│   ├── conftest.py             # Shared fixtures.
│   ├── test_batch_endpoints.py # Cross-cutting batch/mega endpoint tests.
│   ├── test_blur_detector.py   # BlurDetector unit tests.
│   ├── test_blur_classifier.py # BlurClassifier unit tests.
│   ├── test_blur_endpoint.py   # /blur/* API-level tests.
│   ├── test_face_matcher.py    # Cosine similarity / find_matches tests.
│   ├── test_face_endpoints.py  # /faces/* API-level tests.
│   ├── test_bib_recognizer.py  # BibRecognizer unit tests.
│   ├── test_bib_endpoint.py    # /bibs/* API-level tests.
│   ├── test_cpp_extension.py   # Numerical parity: C++ ops vs NumPy fallback.
│   ├── unit/                   # Additional unit-scope tests.
│   ├── integration/            # Integration-scope tests.
│   ├── e2e/                    # End-to-end tests against a running stack.
│   └── fixtures/
│       ├── images/
│       └── embeddings/
│
│
├── benchmarks/
│   └── bench_cpp_vs_python.py  # Timing comparisons for all C++ ops vs Python.
│
│
├── models/                     # === ML MODEL FILES (gitignored) ===
│   ├── manifest.json           # Lists required models with sources/notes.
│   ├── blur_classifier/
│   │   ├── blur_classifier.onnx  # YOLOv8n-cls, 4 classes — 98.68% accuracy.
│   │   └── class_names.json
│   ├── bib_detection/
│   │   └── yolov8n_bib.onnx    # Custom-trained bib detector (Ultralytics).
│   └── models/
│       ├── buffalo_l/          # InsightFace RetinaFace + ArcFace ONNX bundle.
│       └── buffalo_l.zip       # Source bundle.
│
│
├── Training-Images/            # Training datasets (gitignored).
│
│
├── runs/                       # Ultralytics training output (gitignored).
│   ├── classify/blur_cls/      # Blur classifier training artifacts.
│   └── detect/                 # Bib detector training artifacts.
│
│
├── scripts/
│   ├── download_models.py          # Verifies which model files are present.
│   ├── seed_db.py                  # Creates a development API key.
│   ├── benchmark.py                # Performance benchmarks.
│   ├── backup_db.sh                # pg_dump helper.
│   │
│   ├── prepare_blur_dataset.py     # Builds train/val splits for the blur classifier.
│   ├── train_blur_classifier.py    # YOLOv8n-cls fine-tune script.
│   ├── export_blur_classifier.py   # Exports best.pt → ONNX → models/blur_classifier/.
│   │
│   ├── train_bib_detector.py       # Trains the dedicated YOLOv8n bib detector.
│   ├── export_bib_detector.py      # Exports bib detector to ONNX (requires --force
│   │                               # to overwrite).
│   ├── extract_bib_labels.py       # Extracts bib-only labels from combined dataset.
│   │
│   ├── auto_annotate_face_bib.py   # Legacy: auto-annotate with InsightFace + PaddleOCR.
│   ├── train_face_bib_detector.py  # Legacy: combined face+bib YOLO (superseded —
│   │                               # face detection now uses InsightFace directly
│   │                               # because the combined model's bib mAP was weak).
│   └── export_face_bib_detector.py
│
│
└── docs/                       # === DOCUMENTATION (you are here) ===
    ├── README.md                   # Doc index.
    ├── ai-system-overview.md       # Current state of all three pipelines.
    ├── api-reference.md            # Every endpoint + request/response examples.
    ├── architecture.md             # 4-layer design + request / batch flow.
    ├── folder-structure.md         # This file.
    ├── tech-stack.md               # Library choices + rationale.
    ├── cpp-integration.md          # C++ extension build + fallback pattern.
    ├── setup-guide.md              # Local dev setup.
    ├── deployment.md               # Docker, GPU, scaling, health checks.
    ├── security.md                 # Auth, rate limits, validation, privacy.
    ├── maintenance-guide.md        # Operator runbook.
    ├── integration-architecture.md # Responsibility boundary across backends.
    └── integration-contracts.md    # Per-backend call patterns with code examples.
```
