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
- **Each worker loads its own models**: GPU workers can be on different machines
- **One task at a time** per worker (GPU-bound, `prefetch_multiplier=1`)

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

When the server receives SIGTERM (e.g., during deployment):

1. Stop accepting new requests
2. Wait for in-flight requests to complete (up to 30 seconds)
3. Unload ML models (free GPU memory)
4. Close Redis connection
5. Close database connection pool
6. Process exits cleanly

This is handled by FastAPI's lifespan context manager and Uvicorn's signal handling.

---

## Model Loading Strategy

Models are loaded at **build time** (in Docker) or **first startup** (local development):

1. **InsightFace models** (RetinaFace + ArcFace): Downloaded automatically by the InsightFace library on first use. Cached in `~/.insightface/` or the configured MODEL_DIR.

2. **PaddleOCR models**: Downloaded automatically by PaddleOCR on first use. Cached in `~/.paddleocr/`.

3. **YOLOv8 bib detector**: Must be custom-trained on a bib number dataset and placed in `models/bib_detection/yolov8n_bib.onnx`.

At server startup, the ModelRegistry loads all models into memory in parallel (using `asyncio.gather`). This takes 5-15 seconds depending on the machine. The readiness probe will report `false` until loading completes.

---

## Environment-Specific Configuration

| Setting | Development | Staging | Production |
|---|---|---|---|
| `DEBUG` | true | false | false |
| `LOG_LEVEL` | DEBUG | INFO | INFO |
| `WORKERS` | 1 | 2 | 4 |
| `USE_GPU` | false | false | true |
| Swagger UI | Enabled (/docs) | Disabled | Disabled |
| Auth required | Optional | Required | Required |
| Rate limits | Disabled (no Redis) | Enabled | Enabled |
