from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "QuickPitik API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    METRICS_ENABLED: bool = True  # off in dev — instrumentator 7.1.0 crashes on FastAPI 0.141 nested routers
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    SQL_ECHO: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 2
    # Total request body ceiling, enforced by BodySizeLimitMiddleware in main.py.
    # Until 2026-08-14 this was defined and never read, so nothing in the app
    # bounded body size and nginx's client_max_body_size was the only guard.
    #
    # Sized by the ONE caller that can legitimately approach it: the backend's
    # Phase-C mega drain. It posts MAX_BATCH_SIZE (50, per its own
    # application.yml batch.max-size) ORIGINAL photos to /faces/enroll/mega and
    # /bibs/recognize/mega, so its worst case is 50 x the 25 MB MAX_FILE_SIZE
    # ceiling = 1250 MB. The previous 1 GB did not cover that: any 50-photo
    # batch averaging >20.5 MB/photo 413s and the WHOLE indexing batch fails
    # before a byte is read. 2 GB covers it with ~64% headroom. (Measured
    # originals for reference — Training-Images/Sharp_images, n=350: mean
    # 5.06 MB, p90 11.64, max 15.11; so the observed-max drain is 756 MB.)
    #
    # /stream is NOT what sizes this, contrary to the note this comment
    # replaced. The desktop resizes to 1280px/q90 before uploading
    # (BLUR_AI_MAX_DIMENSION in BatchMyPhotos), giving ~253 KB per image and
    # ~49 MB for its 200-image chunk — 40x under the ceiling.
    #
    # Consequence accepted: /stream buffers every file in RAM with no blob-store
    # offload, so its worst case doubles with this value. See docs/security.md.
    #
    # Keep in lockstep with client_max_body_size in nginx.conf — TestBodySizeLimit
    # parses that file and fails if the two drift.
    MAX_REQUEST_BODY: int = 2 * 1024 * 1024 * 1024  # 2 GB

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/quickpitik"

    # Redis (use redis://:password@host:port/db for authenticated instances)
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML Models
    MODEL_DIR: str = "./models"
    USE_GPU: bool = False
    GPU_DEVICE: int = 0

    # Blur Detection
    BLUR_THRESHOLD: float = 100.0
    BLUR_DETECTION_MIN_CONFIDENCE: float = 0.5
    # Resolution the blur CLASSIFIER decodes to before inference. Calibrated, not
    # arbitrary: re-measured 2026-08-14 on 1057 labelled val images after
    # _preprocess switched to INTER_AREA, so this is no longer just standing in
    # for missing anti-aliasing — 640 is genuinely the better input scale.
    # Holding preprocess constant and varying only this value:
    #   640  -> 4/698 blurry called sharp
    #   1280 -> 7/698 blurry called sharp
    # (End-to-end served accuracy is 98.30% — `python scripts/blur_gate.py`.)
    # Shared by /blur/classify, /blur/classify/stream and the blur_classify_batch
    # worker — keep all three on one value or they will disagree on the same
    # photo. Distinct from MAX_INFERENCE_DIMENSION (1280), which is sized for bib
    # OCR and must stay there.
    BLUR_CLASSIFY_DECODE_DIM: int = 640

    # Face Recognition
    FACE_SIMILARITY_THRESHOLD: float = 0.4
    FACE_DET_SIZE: int = 640
    FACE_MIN_ENROLLMENT_CONFIDENCE: float = 0.7

    # Bib Recognition
    BIB_MIN_CHARS: int = 2

    # Inference
    INFERENCE_TIMEOUT: int = 120  # Per-image inference timeout in seconds
    INFERENCE_BATCH_TIMEOUT: int = 300  # Max seconds for any single batch ONNX call
    INFERENCE_SUB_BATCH_SIZE: int = 50  # Match MAX_BATCH_SIZE: 1 ONNX call per task
    # ONNX Runtime thread pools for the blur classifier session. On CPU hosts keep
    # WORKERS x ONNX_INTRA_OP_THREADS <= physical cores to avoid oversubscription
    # (e.g. WORKERS=2 -> set intra=3 on a 6-core box). The streaming classify path
    # now batches per sub-batch, so per-request thread pressure is lower than the
    # old per-image fan-out. On GPU (USE_GPU=true) inference runs on-device and
    # these matter less — batching (dynamic axis) is what drives GPU throughput.
    ONNX_INTRA_OP_THREADS: int = 6
    ONNX_INTER_OP_THREADS: int = 4

    # OCR
    OCR_MAX_WORKERS: int = 8  # Thread pool size for PaddleOCR batch inference

    # Auth
    API_KEY_HEADER: str = "X-API-Key"
    JWT_PUBLIC_KEY: str = ""
    JWT_ALGORITHM: str = "RS256"

    # Rate Limiting
    RATE_LIMIT_DEFAULT: int = 60
    RATE_LIMIT_BURST: int = 10

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # Webhooks
    WEBHOOK_TIMEOUT: int = 10
    WEBHOOK_MAX_RETRIES: int = 3
    WEBHOOK_SECRET_KEY: str = ""  # Fernet key for encrypting webhook secrets at rest

    # Celery
    CELERY_SECURITY_KEY: str = ""

    # File Upload
    # 25 MB matches the Spring backend's multipart ceiling
    # (application.yml spring.servlet.multipart.max-file-size). It forwards the
    # ORIGINAL uploaded bytes to /faces/enroll and /bibs/recognize, so anything
    # it accepts must be accepted here too — at the previous 10 MB, ~1% of real
    # photographer originals uploaded fine and then failed indexing.
    MAX_FILE_SIZE: int = 25 * 1024 * 1024  # 25 MB
    # Upload bound on the longest edge — NOT the decode target (that is
    # MAX_INFERENCE_DIMENSION below, which every path downscales to anyway).
    # This exists only to fail closed on absurd input; it is a bomb guard, not
    # a quality gate, so it is sized to pass any real camera:
    #   largest input measured in Training-Images/ is 6000x4000 (24 MP DSLR);
    #   12000 clears 102 MP medium format (11648x8736) with headroom.
    # The previous hardcoded 4096 rejected 3.75% of sampled real photos on the
    # single-image endpoints while the batch/mega paths accepted them, so the
    # same photo indexed or failed depending on INDEXING_MODE. Enforced by
    # validate_and_decode AND validate_batch_file so the two cannot drift.
    MAX_IMAGE_DIMENSION: int = 12000
    MAX_BATCH_SIZE: int = 50  # Raised from 20: Redis now 1GB + lazy decode keeps memory O(1)
    MEGA_BATCH_MAX_SIZE: int = 500  # Max images per mega-batch request (server-side chunking)
    STREAM_BATCH_MAX_SIZE: int = 500  # Max images per streaming sync batch request
    STREAM_CONCURRENCY: int = 8  # Thread pool size for concurrent image processing
    STREAM_CLASSIFY_MAX_SIZE: int = 500  # Max images per streaming classify request
    STREAM_CLASSIFY_CONCURRENCY: int = 8  # Thread pool for classify/stream (ONNX is heavier)
    MAX_ACTIVE_JOBS_PER_KEY: int = 10  # Backpressure: max pending+processing jobs per API key
    # Internal-tier callers (e.g. the Spring backend's bulk photo-indexing
    # drain) submit a FACE + BIB job per event-batch and may drain several
    # events at once, so they get a higher ceiling than the public default.
    MAX_ACTIVE_JOBS_PER_KEY_INTERNAL: int = 50
    JOB_RETENTION_DAYS: int = 7  # Auto-delete completed/failed jobs older than this

    # Blob store — images written to shared volume instead of base64-in-Redis
    BLOB_STORE_PATH: str = "/tmp/quickpitik-blobs"

    # Image preprocessing — downscale large images before inference.
    # 640 was too aggressive for marathon photos: a 2400x1600 frame got shrunk
    # to 426x640, leaving the bib at ~40x27px which is below the YOLO bib
    # detector's effective range. 1280 keeps inference fast while preserving
    # enough bib pixels for OCR (~80x54px on a typical race shot).
    #
    # Shared by the FACE and BIB pipelines end to end — single endpoints AND
    # Celery workers. The workers used to hardcode their own values (face 768,
    # bib 1024) while the endpoints used this one, so the same photo produced
    # different results depending on which path indexed it. Re-measured
    # 2026-08-14 on the labelled val sets before unifying:
    #   face (323 photos, 655 GT faces): 1280 detects 657 vs 768's 653, and
    #     never fewer; 4 faces are found ONLY at 1280. Embedding drift between
    #     the two dims for the same face is cosine 0.974 mean / 0.818 min —
    #     comfortably above the backend's 0.6 match threshold, so the drift did
    #     not flip matches, but it did mean two embedding populations per event.
    #     Cost of 1280 over 768: +3% end-to-end (+11.7 ms/image); decode is
    #     unchanged (the full-res JPEG decode dominates the resize) and
    #     RetinaFace letterboxes to FACE_DET_SIZE either way.
    #   bib (80 photos): identical YOLO detections (the bib ONNX is a fixed
    #     640 export, so decode dim never reached it) but 17/80 images READ a
    #     DIFFERENT bib string at 1024 vs 1280 — e.g. 8037/6037, 80913/80915.
    #     A bib number is an exact-match key for participant lookup, so that
    #     divergence mis-attributes photos. 1024 was also below the floor this
    #     setting was raised to in the first place.
    # Blur is deliberately NOT on this value: see BLUR_CLASSIFY_DECODE_DIM.
    MAX_INFERENCE_DIMENSION: int = 1280

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
