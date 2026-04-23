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

### Production (CPU)
```bash
docker compose -f docker-compose.yml up --build -d
```
Uses the production `Dockerfile` (multi-stage, optimized, no hot-reload).

### Production (GPU)
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```
Requires NVIDIA Container Toolkit installed on the host. The GPU override adds:
- `runtime: nvidia` device reservation
- `USE_GPU=true` environment variable

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
| Blur detection | ~2ms | N/A | Always CPU (too fast to benefit from GPU) |
| RetinaFace (face detection) | ~80ms | ~8ms | Production with faces |
| ArcFace (face embedding) | ~50ms | ~5ms | Production with faces |
| YOLOv8n (bib detection) | ~40ms | ~4ms | Production with bibs |
| PaddleOCR (bib OCR) | ~30ms | ~6ms | Production with bibs |

**GPU is 10x faster for neural network inference.** But for development and testing, CPU is fine.

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
- If face embeddings exceed ~10M, consider migrating vector search to dedicated Milvus

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

### Readiness: GET /api/v1/health/ready
- Checks: models loaded + database reachable + Redis reachable
- Used by load balancers to know when to send traffic
- If this fails, stop routing traffic to this instance (but don't restart)

### Docker health check
The Dockerfile includes a built-in health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/health').raise_for_status()"
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

Models are loaded at **build time** (in Docker) or **first startup** (local development):

1. **InsightFace models** (RetinaFace + ArcFace): Downloaded automatically by the InsightFace library on first use. Cached in `~/.insightface/` or the configured MODEL_DIR.

2. **PaddleOCR models**: Downloaded automatically by PaddleOCR on first use. Cached in `~/.paddleocr/`.

3. **YOLOv8 bib detector**: Custom-trained ONNX ships in the repo at `models/bib_detection/yolov8n_bib.onnx`. Replace it to use a newer model.
4. **Blur classifier**: Custom-trained ONNX ships in the repo at `models/blur_classifier/blur_classifier.onnx` plus `class_names.json`.

At server startup, the `ModelRegistry` loads all models in parallel (via `asyncio.gather` + `asyncio.to_thread`). Heavy libraries (`torch`, `insightface`, `ultralytics`) are pre-imported on the main thread first to avoid Windows DLL loading races. This takes 5–15 seconds depending on the machine. The readiness probe returns 503 until loading completes.

---

## Environment-Specific Configuration

| Setting | Development | Staging | Production |
|---|---|---|---|
| `DEBUG` | `true` | `false` | `false` (startup aborts if `true` + `ENVIRONMENT=production`) |
| `ENVIRONMENT` | `development` | `staging` | `production` |
| `LOG_LEVEL` | `DEBUG` | `INFO` | `INFO` |
| `WORKERS` | 1–2 | 2 | scale by CPU |
| `USE_GPU` | `false` | `false` | `true` (needs `[gpu]` extras + CUDA) |
| Swagger UI / `/redoc` | Enabled | Disabled | Disabled |
| API-key auth | Bypassable when `DEBUG=true` | Required | Required |
| `/metrics` | Open (no auth) | Auth required | Auth required |
| Rate limits | Enforced if Redis is up; no-op otherwise | Enforced | Enforced |
| `WEBHOOK_SECRET_KEY` | Optional (plaintext fallback) | Set | Set |
| HSTS header | Off | Off | On (`strict-transport-security`) |
| `BLOB_STORE_PATH` | `/tmp/eventai-blobs` | Shared volume | Shared volume mounted in API + workers |
