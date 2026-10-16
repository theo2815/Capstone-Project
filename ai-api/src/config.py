from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# DEBUG turns the API-key check off (src/middleware/auth.py), so it is allowed only
# when ENVIRONMENT was set on purpose to one of these. The "development" default does
# not count: a deployment that forgets ENVIRONMENT must not run with the check off.
DEV_ENVIRONMENTS = frozenset({"development", "local", "test"})


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "QuickPitik API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    METRICS_ENABLED: bool = True
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    SQL_ECHO: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = Field(default=2, gt=0)
    READINESS_TIMEOUT_SECONDS: float = Field(default=3.0, gt=0)
    BLUR_REQUEST_TIMEOUT_SECONDS: float = Field(default=60, gt=0)
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
    # /stream now retains one sub-batch; the dedicated blur profile has its own
    # smaller transport and decode budget. See docs/security.md.
    #
    # Keep in lockstep with client_max_body_size in nginx.conf — TestBodySizeLimit
    # parses that file and fails if the two drift.
    MAX_REQUEST_BODY: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)  # 2 GB
    # Dedicated 2-GiB blur service: two admitted requests can spool at most
    # 256 MiB, while decoding is capped separately below.
    BLUR_MAX_REQUEST_BODY: int = Field(default=128 * 1024 * 1024, gt=0)
    BLUR_MAX_IMAGE_PIXELS: int = Field(default=24_000_000, gt=0)
    BLUR_MAX_ACTIVE_REQUESTS: int = Field(default=2, gt=0)
    BLUR_DECODE_CONCURRENCY: int = Field(default=2, gt=0)
    # Stage-2 photos scored at the same time inside one stream sub-batch. 1 = one after
    # another (today). 2 uses both native inference slots of this process for one
    # request: set BLUR_MAX_ACTIVE_REQUESTS=1 beside it, or a second request is still
    # admitted and either stream can then end part-way, marked incomplete.
    BLUR_STAGE2_CONCURRENCY: int = Field(default=1, ge=1, le=2)

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/quickpitik"

    # Redis (use redis://:password@host:port/db for authenticated instances)
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML Models
    MODEL_DIR: str = "./models"
    MODEL_PROFILE: Literal["all", "blur"] = "blur"
    USE_GPU: bool = False
    GPU_DEVICE: int = 0

    # Blur Detection
    BLUR_ASYNC_ENABLED: bool = True
    BLUR_THRESHOLD: float = Field(default=100.0, ge=1.0, le=10000.0)
    # One decode scale for classical single, stream, and worker scores.
    # Re-evaluate labeled images before changing it; BLUR_THRESHOLD is scale-sensitive.
    BLUR_DETECT_DECODE_DIM: int = Field(default=640, gt=0)
    BLUR_DETECTION_MIN_CONFIDENCE: float = Field(default=0.5, ge=0, le=1)
    # Resolution the blur CLASSIFIER decodes to before inference: the full desktop
    # upload (BLUR_AI_MAX_DIMENSION = 1280 in BatchMyPhotos). The two-stage
    # classifier makes its own smaller views from it (640 gray for the whole-frame
    # rule, a 640 padded square for the model), so nothing is thrown away here.
    # The old 640 value was calibrated for the retired 224-px centre-crop model.
    # Shared by /blur/classify, /blur/classify/stream and the blur_classify_batch
    # worker — keep all three on one value or they will disagree on the same
    # photo. Equal to MAX_INFERENCE_DIMENSION today, but a separate setting: that
    # one is the generic colour-decode cap.
    BLUR_CLASSIFY_DECODE_DIM: int = Field(default=1280, gt=0)
    # Whole-frame rule, stage 1 of the blur classifier (src/ml/blur/whole_frame.py).
    # Learned from the labelled event folds by scripts/fit_blur_whole_frame_rule.py:
    # re-fit after the labelled photos change, never hand-tune.
    #   fine detail <= SOFT_MAX_DETAIL and no sideways smear  -> whole-frame soft
    #   direction ratio > MOTION_MIN_RATIO and fine detail <= MOTION_MAX_DETAIL -> motion
    BLUR_SOFT_MAX_DETAIL: float = Field(default=34.1, gt=0)
    BLUR_MOTION_MIN_RATIO: float = Field(default=2.63, gt=0)
    BLUR_MOTION_MAX_DETAIL: float = Field(default=52.4, gt=0)

    # Inference
    INFERENCE_TIMEOUT: int = 120  # Per-image inference timeout in seconds
    INFERENCE_BATCH_TIMEOUT: int = 300  # Max seconds for any single batch ONNX call
    BLUR_INFERENCE_TIMEOUT_SECONDS: float = Field(default=30, gt=0, le=59)
    BLUR_STREAM_TIMEOUT_SECONDS: float = Field(default=120, gt=0)
    INFERENCE_SUB_BATCH_SIZE: int = Field(default=50, gt=0)
    # ONNX Runtime thread pools for the blur classifier session. On CPU hosts keep
    # WORKERS x ONNX_INTRA_OP_THREADS <= physical cores to avoid oversubscription
    # (e.g. WORKERS=2 -> set intra=3 on a 6-core box). The streaming classify path
    # now batches per sub-batch, so per-request thread pressure is lower than the
    # old per-image fan-out. On GPU (USE_GPU=true) inference runs on-device and
    # these matter less — batching (dynamic axis) is what drives GPU throughput.
    ONNX_INTRA_OP_THREADS: int = 6
    ONNX_INTER_OP_THREADS: int = 4
    # Whether ONNX Runtime's worker threads keep spinning between pieces of work. Spinning
    # is fastest on dedicated cores; inside a CPU quota (a 2-vCPU container) it can burn
    # the quota and slow everything else down. False is the setting to try on staging.
    ONNX_ALLOW_SPINNING: bool = True

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
    # Limits a whole webhook delivery, not one read.
    WEBHOOK_DELIVERY_DEADLINE_SECONDS: float = Field(default=30, gt=0)

    # Celery
    CELERY_SECURITY_KEY: str = ""
    CELERY_VISIBILITY_TIMEOUT: int = Field(default=4200, gt=3600)
    JOB_LEASE_SECONDS: int = Field(default=3900, gt=3600)

    # File Upload
    # 25 MB matches the Spring backend's multipart ceiling
    # (application.yml spring.servlet.multipart.max-file-size). It forwards the
    # ORIGINAL uploaded bytes to /faces/enroll and /bibs/recognize, so anything
    # it accepts must be accepted here too — at the previous 10 MB, ~1% of real
    # photographer originals uploaded fine and then failed indexing.
    MAX_FILE_SIZE: int = Field(default=25 * 1024 * 1024, gt=0)  # 25 MB
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
    MAX_IMAGE_DIMENSION: int = Field(default=12000, gt=0)
    MAX_BATCH_SIZE: int = Field(default=50, gt=0)
    MEGA_BATCH_MAX_SIZE: int = Field(default=500, gt=0)
    STREAM_BATCH_MAX_SIZE: int = Field(default=500, gt=0)
    STREAM_CONCURRENCY: int = Field(default=8, gt=0)
    STREAM_CLASSIFY_MAX_SIZE: int = Field(default=500, gt=0)
    STREAM_CLASSIFY_CONCURRENCY: int = Field(default=8, gt=0)
    MAX_ACTIVE_JOBS_PER_KEY: int = 10  # Backpressure: max pending+processing jobs per API key
    # Internal-tier callers (e.g. the Spring backend's bulk photo-indexing
    # drain) submit a FACE + BIB job per event-batch and may drain several
    # events at once, so they get a higher ceiling than the public default.
    MAX_ACTIVE_JOBS_PER_KEY_INTERNAL: int = 50
    JOB_RETENTION_DAYS: int = 7  # Auto-delete completed/failed jobs older than this

    # Local development falls back to disk. Railway async uses a private S3 bucket.
    BLOB_STORE_PATH: str = "/tmp/quickpitik-blobs"
    AWS_S3_BUCKET_NAME: str = ""
    AWS_ENDPOINT_URL: str = ""
    AWS_DEFAULT_REGION: str = "auto"
    AWS_S3_URL_STYLE: str = "virtual"
    BLOB_RETENTION_SECONDS: int = Field(default=86400, gt=0)

    # Image preprocessing — downscale large images before inference.
    # Longest side validate_and_decode shrinks a colour image to (the default of
    # downscale_for_inference). 1280 was measured for the face and bib pipelines,
    # removed 2026-10-16 (the measurements are in git history). /blur/classify
    # reads it only as a first cap; its real decode target is
    # BLUR_CLASSIFY_DECODE_DIM, which the batch worker passes directly.
    MAX_INFERENCE_DIMENSION: int = Field(default=1280, gt=0)

    @model_validator(mode="after")
    def check_upload_budgets(self):
        if self.MODEL_PROFILE == "blur":
            # One API process on the initial 2-vCPU blur service; explicit env
            # values remain available after load measurements justify tuning.
            for name, value in (("WORKERS", 1), ("ONNX_INTRA_OP_THREADS", 2),
                                ("ONNX_INTER_OP_THREADS", 1)):
                if name not in self.model_fields_set:
                    setattr(self, name, value)
            if self.METRICS_ENABLED and self.WORKERS != 1:
                raise ValueError(
                    "blur metrics require WORKERS=1 until multiprocess aggregation exists"
                )
        if self.MAX_REQUEST_BODY <= self.MAX_FILE_SIZE:
            raise ValueError("MAX_REQUEST_BODY must exceed MAX_FILE_SIZE")
        if self.BLUR_MAX_REQUEST_BODY <= self.MAX_FILE_SIZE:
            raise ValueError("BLUR_MAX_REQUEST_BODY must exceed MAX_FILE_SIZE")
        if self.BLUR_MAX_IMAGE_PIXELS > self.MAX_IMAGE_DIMENSION**2:
            raise ValueError("BLUR_MAX_IMAGE_PIXELS must fit MAX_IMAGE_DIMENSION")
        if self.MODEL_PROFILE == "blur":
            if self.BLUR_MAX_REQUEST_BODY * self.BLUR_MAX_ACTIVE_REQUESTS > 256 * 1024 * 1024:
                raise ValueError("blur admitted upload bytes exceed the 256 MiB budget")
            if self.BLUR_MAX_IMAGE_PIXELS * self.BLUR_DECODE_CONCURRENCY > 48_000_000:
                raise ValueError("blur simultaneous decode pixels exceed the 48 MP budget")
            if self.INFERENCE_SUB_BATCH_SIZE > 50:
                raise ValueError("blur inference sub-batches must contain at most 50 images")
        return self

    @model_validator(mode="after")
    def refuse_debug_outside_dev(self):
        if self.DEBUG and (
            "ENVIRONMENT" not in self.model_fields_set
            or self.ENVIRONMENT not in DEV_ENVIRONMENTS
        ):
            raise ValueError(
                "DEBUG=true is only allowed when ENVIRONMENT is set to one of "
                f"{', '.join(sorted(DEV_ENVIRONMENTS))}; set DEBUG=false otherwise"
            )
        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        # A settings error at startup must not print env values (secrets) into logs.
        "hide_input_in_errors": True,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
