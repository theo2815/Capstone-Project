# Deployment

## Local Development

```bash
# Start database and cache
docker compose up db redis -d

# Run the API with auto-reload
make dev
```

## Docker (Full Stack)

### Development
```bash
docker compose up --build
```
Starts 4 containers:
- `ai-api` (port 8000) - FastAPI with hot-reload, code mounted as volume
- `celery-worker` - Background task processor
- `db` (port 5432) - PostgreSQL 16 with pgvector
- `redis` (port 6379) - Cache and task queue broker

> **`docker-compose.yml` is the DEV stack and must never be used to deploy.** It builds
> `Dockerfile.dev` and sets `DEBUG=true` + `ENVIRONMENT=development`, which means API-key auth
> is **bypassed with full `["*"]` scope** (`middleware/auth.py`), Swagger/ReDoc are served, and
> `/metrics` is open. This section of the doc previously named it as the production command.

### Production prerequisites

Production reads two values that have no safe default. Put them in `.env` next to
`docker-compose.prod.yml`, or export them:

| Variable | Why |
|---|---|
| `POSTGRES_PASSWORD`, `REDIS_PASSWORD` | Datastore credentials. |
| `ALLOWED_ORIGINS` | **Required — the stack refuses to start without it.** Set your real origins, e.g. `["https://quickpitik.com"]`. It used to default to `["*"]`. |
| `WEBHOOK_SECRET_KEY` | Fernet key encrypting webhook secrets at rest. Empty = plaintext (the app only warns). Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Must be the **same value across the API and all four workers** — the API encrypts and the blur/face/bib workers decrypt, so a mismatch breaks every signature. |

Model weights are **mounted from the host**, not baked into the image — confirm `./models` is
populated before starting (see "Model Loading Strategy" below):

```bash
python scripts/download_models.py    # exits non-zero and names anything missing
```

### Production (CPU)
```bash
docker compose -f docker-compose.prod.yml up --build -d
```
Uses the production `Dockerfile` (multi-stage, optimized, no hot-reload). This is a complete
standalone stack — API, four per-queue Celery workers, db, redis, nginx and certbot. The API
publishes on `127.0.0.1:8000` only; public traffic arrives through nginx on 80/443, so the
proxy's rate limit and `client_max_body_size` cannot be bypassed.

The Dockerfile defaults to `MODEL_PROFILE=all` and installs the full application dependencies.
For a smaller CPU blur image, supply the audited ONNX and ordered labels under
`models/blur_classifier/` in the trusted build workspace, then build with
`docker build --build-arg MODEL_PROFILE=blur -t quickpitik-blur:2026-08-14-dynamic .` and run it
with `MODEL_PROFILE=blur`. This profile installs the pinned
`requirements-blur.lock` and loads only blur models; face and bib routes return 404.
For the initial dedicated Railway blur launch, `.railway/railway.ts` sets
`BLUR_ASYNC_ENABLED=false`. The shared all-model service keeps the default
`true`. The desktop calls `/api/v1/blur/classify/stream` and falls back to
`/api/v1/blur/classify`; both remain available.

**Initial Railway blur launch checklist:**
- Link the staging directory to the isolated blur project, then inspect
  `railway config plan`: it must name `quickpitik-ai-api-blur` and change only
  `ai-api-blur.BLUR_ASYNC_ENABLED`. Abort if it names the backend project.
- Verify `BLUR_ASYNC_ENABLED=false` on `ai-api-blur` before exposing it. The
  following POST routes return 503 `BLUR_ASYNC_DISABLED` before multipart parsing:
  `/api/v1/blur/detect/batch`, `/api/v1/blur/classify/batch`,
  `/api/v1/blur/detect/mega`, `/api/v1/blur/classify/mega`.
- Deploy the API with PostgreSQL and Redis for keys, jobs and rate limiting;
  omit the Celery worker, beat scheduler and shared blob volume. Public
  `/api/v1/health/readyz` checks the dependencies needed for synchronous blur;
  it does not certify async job processing.
- Keep R10/R11 async storage, dispatch and replay guarantees open. Complete
  Task 10B and its cross-service tests before setting the switch to `true`
  on this Railway service.

### Optional durable async blur topology (Task 10B)

Keep `.railway/railway.ts` as the stream-only default. When async batch/mega is
approved for a separate release, preview `.railway/railway-async.ts` with
`railway config plan --file .railway/railway-async.ts` from an isolated staging
directory linked to **quickpitik-ai-api-blur**. Review resource creation and
deletion before applying it. That plan adds one private Railway Bucket, one
blur worker (`blur,default`, concurrency 1), and one beat scheduler; it enables
async on the API. Deploy the same verified `MODEL_PROFILE=blur` image/source to
all three services. Run `railway up . --path-as-root --no-gitignore --service
<service-name>` separately for the API, worker and beat from the trusted build
staging directory after the configuration is applied. Deploy the migration
before accepting async traffic. The worker and beat have no public domain.

The bucket's `BUCKET`, `ENDPOINT`, `ACCESS_KEY_ID`, `SECRET_ACCESS_KEY` and
`REGION` references feed the matching `AWS_*` variables in each service. Use
the bucket's reported URL style if it differs from `virtual`. Do not share its
credentials with desktop keys. The API uploads opaque `jobs/<uuid>/...` keys
before dispatch, and the worker reads at most the configured per-file limit.
The single beat process runs the reaper every 5 minutes and blob cleanup every
30 minutes. Terminal inputs remain for `BLOB_RETENTION_SECONDS` (default 24 h)
after completion; orphan uploads use the same age window. A cleanup error is
logged and retried on the next beat pass. Railway Buckets currently have no
native lifecycle policy, so this scheduler is required for retention.

For the existing Redis service, verify `CONFIG GET appendonly appendfsync
maxmemory-policy` returns `yes`, `everysec` and `noeviction`, and confirm its
data volume actually persists through a service restart. Set a memory limit
with headroom for pending chord results and monitor its usage; an evicting
cache policy can silently lose jobs. Celery uses a 4,200-second visibility
timeout, 3,600-second hard task limit and 3,900-second processing lease.
Start with one 2 vCPU / 2 GiB API process and one 2 vCPU / 2 GiB worker with
prefetch 1; measure real queue and memory use before changing either budget.
Keep `WEBHOOK_SECRET_KEY`, Postgres and Redis references aligned across the
services. A successful `/readyz` checks API dependencies, so also run a
restricted-key batch request through the separate worker and confirm the
terminal job before exposing async routes.

The initial one-worker blur budget admits two uploads per process, each at most
128 MiB total, 25 MiB per file and 500 files per stream. Header dimensions
must fit 12,000px per edge and 24 megapixels; at most two images decode at
once across admitted requests. Streams retain one 50-image sub-batch in memory.
The upload contract remains JPEG/PNG/WebP; clients convert HEIF before upload.
The full `all` profile retains its 2 GiB shared body ceiling and existing
per-file limits. Tune the blur values in `src/config.py` only after measuring
real desktop payloads and container peak RSS under the target Railway quota.
Regenerate the lock from `requirements-blur.in` with `pip-compile` in a clean Python 3.12
Linux environment with `libpq-dev` installed. The two files are gitignored; the
build checks their manifest hashes, label order, ONNX shape, and warm inference,
then embeds them in a read-only image directory. No model mount is needed at
runtime. The built image is the immutable artifact containing both files. The
initial Railway blur service was deployed by CLI upload from an isolated staging
directory: a committed `ai-api` source tree plus only the audited ONNX and
`class_names.json` fetched from the private artifact store. Use
`railway up . --path-as-root --no-gitignore --service ai-api-blur` from that
staging directory after linking it to the intended project. For example, from
the monorepo root in PowerShell:

```powershell
$stage = Join-Path $env:TEMP ("quickpitik-blur-" + [guid]::NewGuid())
$projectId = '<isolated-project-id>'
if (-not $env:BLUR_MODEL_URL -or -not $env:BLUR_LABELS_URL) {
    throw 'Set private BLUR_MODEL_URL and BLUR_LABELS_URL first'
}
New-Item -ItemType Directory -Force $stage | Out-Null
git archive --format=tar -o "$stage\source.tar" HEAD:ai-api
tar -xf "$stage\source.tar" -C $stage
Remove-Item -LiteralPath "$stage\source.tar"
New-Item -ItemType Directory -Force "$stage\models\blur_classifier" | Out-Null
curl.exe -fsSL $env:BLUR_MODEL_URL -o "$stage/models/blur_classifier/blur_classifier.onnx"
if ($LASTEXITCODE -ne 0) { throw 'Model fetch failed' }
curl.exe -fsSL $env:BLUR_LABELS_URL -o "$stage/models/blur_classifier/class_names.json"
if ($LASTEXITCODE -ne 0) { throw 'Labels fetch failed' }
Set-Location $stage
railway link --project $projectId --environment production --service ai-api-blur
railway up . --path-as-root --no-gitignore --service ai-api-blur
```

Use a fresh empty staging directory for each release. The image build
rejects wrong hashes, labels, shape, or warm inference; record its Railway
image digest for each release. Keep `.env`, training data and checkpoints out
of staging. A clean Git checkout alone cannot rebuild this image because the
weights are ignored; private artifact access is required for a new build.

The blur service's `.railway/railway.ts` sets the pre-deploy migration and
`/api/v1/health/readyz` healthcheck. Run `npm ci --prefix .railway`, then
`railway config plan` and `railway config apply` from the staging directory
linked to the isolated blur project when those settings change. Check the plan
for deletions before applying it. Its `preserve()` entries retain the existing
Railway variables, including secrets; their values do not enter source control.
With npm-installed Railway CLI 5.49.1 on Windows, set `$env:_` to its
`@railway/cli/bin/railway.exe` path before `config plan`/`apply`; otherwise the
SDK can misread the CLI version.

The configured Railway pre-deploy command runs `python -m alembic upgrade head`
with `DATABASE_URL` set to a `postgresql+asyncpg://` URL for a database that permits the
`vector` extension. The existing Alembic chain creates that extension; abort
the deploy if migrations fail. Provision the desktop key through the existing
workflow in the gitignored `../docs/api-keys.md`, with only `blur:read` and
`jobs:read`; the isolated Railway database uses that existing restricted
desktop key. Do not run `scripts/seed_db.py` in staging or production. Railway
injects `PORT`; set `WORKERS` for the service (1 on the initial blur service).
Probe `/api/v1/health/readyz` for public 200/503 readiness; `/api/v1/health` is
liveness, and `/api/v1/health/ready` remains authenticated and detailed.

The blur profile defaults to one API process and 2/1 ONNX intra/inter threads.
`WORKERS=1` is required while metrics are enabled; adding workers needs
Prometheus multiprocess aggregation first. `src/config.py` owns
the request, stream, and per-inference deadlines. Blur request admission is
bounded before multipart parsing; native inference has two process-wide slots
and **zero queue wait**. Exhaustion returns 503 `BLUR_BUSY` with `Retry-After: 1`.
The separate per-key rate limit uses 429. A timed-out native call may still run:
its slot stays occupied, and readiness is 503 until it exits. A stream deadline
ends with `complete: false`; clients must retry unaccounted indices. Railway's
deployment health check is not continuous monitoring, so an instance stuck in
native code needs an operational restart after it becomes unready.

### Railway operations and release check

Use the dedicated `MODEL_PROFILE=blur`, `WORKERS=1`, `METRICS_ENABLED=true`,
`DEBUG=false` profile. Railway supplies `PORT`; the Docker command and health
check use it. Configure a TLS domain and keep `/api/v1/health/readyz` public for
the Railway health check. Set `DATABASE_URL`, `REDIS_URL`,
`ALLOWED_ORIGINS`, and `WEBHOOK_SECRET_KEY` as Railway variables. The
`.railway/railway.ts` pre-deploy command runs Alembic; verify its success and
pgvector availability before routing traffic. Provision a restricted desktop key
using the private `../docs/api-keys.md` workflow and a separate
`metrics:read` operations key. Never use `scripts/seed_db.py` in production.

The initial service exposes single-image and stream blur only. No worker, beat,
or blob volume is needed while `BLUR_ASYNC_ENABLED=false`. The optional async
topology above requires its private bucket, worker, beat, Redis persistence,
and `BLOB_RETENTION_SECONDS` cleanup. Keep one API process until a measured
capacity review. Record the chosen Railway CPU, RAM, monthly budget, domain,
availability alert, termination grace, deployment overlap, image digest, model
digest, and previous working deployment before rollout. The 2-vCPU/2-GiB local
measurement in `docs/security.md` is a starting profile, not a Railway promise.

GitHub Actions `ai-api-blur` runs deterministic regressions on pushes and PRs.
The manual `release-check` job builds from a clean checkout after fetching
private ONNX, labels, and an approved labelled holdout through the
`blur-release` environment's `BLUR_MODEL_URL`, `BLUR_LABELS_URL`, and
`BLUR_HOLDOUT_URL` secrets. The holdout ZIP holds one folder per event, each
with class directories (`<event-id>/sharp/`, `<event-id>/motion_blurred/`, …;
see `scripts/blur_gate.py --help`); the gate reports each event separately and
passes on sharp-vs-blurry results. Missing artifacts fail. Set these approved
limits in that environment's variables: `BLUR_MIN_ACCURACY`,
`BLUR_MAX_MISS_RATE`, `BLUR_MAX_OVERCULL_RATE`, `BLUR_MAX_PHOTOMETRIC_FLIP_RATE`
(the most decisions that brighter, darker or grayscale copies may change), and
the holdout floor `BLUR_MIN_SHARP`, `BLUR_MIN_BLUR`, `BLUR_MIN_EVENTS`. The
approved values are the owner decisions in the vault blur recovery plan
(2026-09-29, D3 and D5). Missing or invalid limits fail before the
release smoke creates database keys. It migrates disposable
PostgreSQL/pgvector, starts Redis and the built image with async routes disabled,
checks real inference, verifies batch/mega requests return `BLUR_ASYNC_DISABLED`,
denies desktop access to `/metrics`, and runs the complete `blur_gate.py`
holdout gate. The private report is retained for seven days. No deployment
follows automatically. The independent holdout and acceptance limits remain
the Task 8 prerequisite; the existing night-event sharp sample already shows
over-culling (see `docs/ai-system-overview.md`).

Scrape `/metrics` with only the operations key. Watch `blur_images_total`
(attempted/succeeded/failed/unaccounted), `blur_rejected_requests_total`,
`blur_active_requests`, `blur_timeouts_total`, `blur_first_result_seconds`,
`blur_stream_duration_seconds`, `blur_streams_total`, `blur_model_info`,
`blur_dependency_ready`, and the Prometheus `process_resident_memory_bytes`
collector. Admission has no waiting queue, so busy rejections measure
saturation. Read logs by request ID; stream completion logs include counts and
outcome, with no image bytes or full filenames. Roll back to the recorded
deployment if readiness stays 503, native timeouts persist, image errors or
unaccounted results rise, or memory approaches the allocated limit. Scale only
after measuring the chosen quota and a drain/redeploy test in Task 12.

### Production (GPU)
```bash
docker compose -f docker-compose.prod.yml -f docker-compose.gpu.yml up --build -d
```
Requires NVIDIA Container Toolkit installed on the host. The GPU overlay adds an `nvidia`
device reservation and `USE_GPU=true` to `ai-api`, `celery-blur`, `celery-face` and
`celery-bib`. `celery-default` stays on CPU — it runs webhook and maintenance tasks and loads
no models. Layer it on `docker-compose.prod.yml`, never on `docker-compose.yml`.

---

## How the Docker Images Work

### Production Dockerfile (multi-stage)

```
Stage 1 (builder):
  - Starts from python:3.12-slim
  - Installs all Python dependencies into /install
  - This stage is thrown away after build

Stage 2 (runtime):
  - Starts from a clean python:3.12-slim
  - Copies only the installed packages from Stage 1
  - Copies application code
  - Result: smaller image (no build tools, no cache)
```

### Development Dockerfile

- Single stage with all dev dependencies
- Source code mounted as a Docker volume (changes reflected instantly)
- Uvicorn runs with `--reload` flag

---

## GPU vs CPU Inference

| Model | CPU Latency | GPU Latency | When to Use GPU |
|---|---|---|---|
| Blur detection | Measure on the chosen quota | Measure if GPU is selected | CPU for the initial blur profile |
| Face, bib and OCR | Measure per workload | Measure per workload | Separate all-model profile |

No generic speed multiplier or latency target is certified for Railway.

The code handles this automatically via ONNX Runtime's provider selection:
```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# If CUDA is available, uses GPU. If not, falls back to CPU.
```

---

## Horizontal Scaling

```
                  Load Balancer (nginx / cloud LB)
                  /           |           \
            [API Pod 1]  [API Pod 2]  [API Pod 3]
                  \           |           /
                   +-- Redis Cluster --+
                   +-- PostgreSQL -----+
                  /           |
        [Celery Worker 1] [Celery Worker 2]
```

### API pods
- **Stateless**: Each pod loads models into its own memory at startup (~1.5GB)
- **Scale by adding pods**: More pods = more concurrent requests
- **No shared state**: All state is in PostgreSQL and Redis

### Celery workers
- **Scale independently**: Add workers based on queue depth
- **Each worker loads its own models**: set `WORKER_QUEUES=blur` (or `face`, `bib`) and start the worker with `-Q blur` to load only what that worker serves. Dev workers without `WORKER_QUEUES` load everything.
- **Prefetch = 4**: `worker_prefetch_multiplier=4` is safe because task messages carry file paths (blob store), not base64 image bytes.
- **Time limits**: `task_soft_time_limit=3300s`, `task_time_limit=3600s`. `worker_max_tasks_per_child=500`, `worker_max_memory_per_child=2GB`.

### Database
- PostgreSQL scales vertically first (bigger machine)
- Add read replicas for search-heavy workloads
- If face embeddings exceed ~10M, consider migrating vector search to dedicated Qdrant

### Redis
- Single node is fine for startup scale
- Redis Sentinel for high availability
- Redis Cluster for horizontal scaling

---

## Health Checks

### Liveness: GET /api/v1/health
- Returns 200 if the process is alive
- Used by container orchestrators to detect crashed processes
- If this fails, restart the container

### Public readiness: GET /api/v1/health/readyz
- Returns only `{"status":"ready"}` (200) or `{"status":"unavailable"}` (503)
- Checks the profile's required models, database and Redis with bounded probes
- Use this path for container and Railway readiness checks

### Authenticated detail: GET /api/v1/health/ready
- Returns the model breakdown and dependency status to an authorized API key

The readiness payload also carries a per-model breakdown:

```json
{"models_loaded": true, "database": true, "redis": true,
 "models": {"blur": true, "face": true, "bib_ocr": true,
            "blur_classifier": false, "bib_detector": false}}
```

In the full profile, `models_loaded` covers `blur`, `face`, and `bib_ocr`; the
classifier remains optional. In the blur profile, `blur` and `blur_classifier`
are both required, so a missing or invalid classifier makes readiness 503.

### Docker health check
The Dockerfile includes a built-in health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/v1/health/readyz" || exit 1
```

---

## Graceful Shutdown

When the server receives SIGTERM (e.g., during deployment) the FastAPI lifespan context shuts down in order:

1. Stop accepting new requests (uvicorn)
2. Brief 2-second drain for in-flight requests
3. `registry.unload_all()` — clears ONNX sessions, InsightFace models, PaddleOCR engine, Ultralytics YOLO models
4. Close Redis connection (`redis.aclose()`)
5. Close the database connection pool
6. Process exits

The 2-second drain is intentionally short — reverse proxies / load balancers are expected to stop routing to the instance before SIGTERM.

---

## Model Loading Strategy

**Full-profile weights are mounted; the dedicated blur image carries its audited pair.**
`models/*` is gitignored (only `manifest.json` is tracked), so a clean Git checkout has no
weights. `docker-compose.prod.yml` mounts the real `./models` directory read-only at
`/app/models` for the full-profile API and workers. Miss that mount and the full-profile
container starts with no weights at all.

1. **InsightFace models** (RetinaFace + ArcFace): downloaded automatically on first use into
   `MODEL_DIR` — so with the mount in place they are already there, and with it missing every
   container re-downloads ~300 MB on every deploy (or fails outright with no egress). Note the
   doubled path: `FaceAnalysis(root=MODEL_DIR)` appends its own `models/`, so the bundle lives
   at `models/models/buffalo_l/`.

2. **PaddleOCR models**: downloaded automatically by PaddleOCR on first use, cached outside
   `models/` (`~/.paddleocr` or `PADDLE_PDX_MODEL_SOURCE`). Nothing to mount.

3. **YOLOv8 bib detector**: custom-trained ONNX at `models/bib_detection/yolov8n_bib.onnx`,
   produced by `scripts/train_bib_detector.py` → `scripts/export_bib_detector.py`. **Optional**
   — without it, bib recognition falls back to full-image OCR with more false positives.

4. **Blur classifier**: custom-trained ONNX at `models/blur_classifier/blur_classifier.onnx`
   plus `class_names.json`. Required by the dedicated blur image; optional in the full profile.

For the full profile, the classifier and bib detector remain optional to the registry. Run
`python scripts/download_models.py` before deploying: it checks every artifact named in
`models/manifest.json` and exits non-zero listing whatever is absent.

At server startup, the `ModelRegistry` loads selected models in parallel (via `asyncio.gather` + `asyncio.to_thread`). In the full profile, heavy libraries (`torch`, `insightface`, `ultralytics`) are pre-imported on the main thread to avoid Windows DLL loading races. The server accepts requests after startup; `/api/v1/health/readyz` returns 503 if required inference, PostgreSQL, or Redis is unavailable.

---

## Environment-Specific Configuration

| Setting | Development | Staging | Production |
|---|---|---|---|
| `DEBUG` | `true` | `false` | `false` (startup aborts if `true` + `ENVIRONMENT=production`) |
| `ENVIRONMENT` | `development` | `staging` | `production` |
| `LOG_LEVEL` | `DEBUG` | `INFO` | `INFO` |
| `WORKERS` | 1–2 | 1 for blur | 1 for blur while metrics are enabled |
| `USE_GPU` | `false` | `false` | `false` for the initial blur profile |
| Swagger UI / `/redoc` | Enabled | Disabled | Disabled |
| API-key auth | Bypassable when `DEBUG=true` | Required | Required |
| `/metrics` | Open when `DEBUG=true` | Explicit `metrics:read` key | Explicit `metrics:read` key |
| Rate limits | 503 when Redis is unavailable | 503 when Redis is unavailable | 503 when Redis is unavailable |
| `WEBHOOK_SECRET_KEY` | Optional (plaintext fallback) | Set | Set |
| HSTS header | Off | Off | On (`strict-transport-security`) |
| `BLOB_STORE_PATH` | `/tmp/quickpitik-blobs` | Unused for stream-only blur | Unused for stream-only blur |
