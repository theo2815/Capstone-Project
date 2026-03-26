# EventAI AI-API Maintenance Guide

**Last updated:** 2026-03-26
**For:** Agents and operators managing the EventAI AI-API in production

---

## 1. System Overview

EventAI AI-API is a FastAPI-based computer vision service for marathon/running event photography. It provides three ML pipelines:

| Pipeline | Purpose | Models Used | Endpoint Prefix |
|----------|---------|-------------|-----------------|
| **Blur Detection** | Filter blurry photos | Laplacian/FFT detector + YOLOv8n-cls ONNX classifier | `/api/v1/blur/` |
| **Face Search** | Match runners across photos | InsightFace (RetinaFace + ArcFace) + pgvector | `/api/v1/faces/` |
| **Bib Number OCR** | Read bib numbers from photos | YOLOv8n detector + PaddleOCR 3.x (PP-OCRv5) | `/api/v1/bibs/` |

### Architecture

```
Client -> FastAPI (uvicorn) -> ML Inference (sync, via asyncio.to_thread)
                            -> PostgreSQL 16 + pgvector (embeddings, jobs, persons)
                            -> Redis (rate limiting, API key cache)
                            -> Celery Worker (batch jobs, webhooks)
```

### Key Files

| File | Role |
|------|------|
| `src/main.py` | App factory, middleware stack, lifespan |
| `src/config.py` | All settings (`Settings` class, `.env` driven) |
| `src/ml/registry.py` | Model loading/unloading at startup |
| `src/workers/celery_app.py` | Celery config, task routing, beat schedule |
| `src/workers/model_loader.py` | Worker-side model loading on fork |
| `src/db/session.py` | Async DB engine + session management |
| `src/db/sync_session.py` | Sync DB engine for Celery workers |

---

## 2. Routine Maintenance Tasks

### Daily

| Task | How | What to Check |
|------|-----|---------------|
| Check health endpoint | `GET /api/v1/health` | Returns `{"status": "alive"}` with HTTP 200 |
| Check readiness | `GET /api/v1/health/ready` (requires API key) | `models_loaded`, `database`, `redis` all true; HTTP 200 |
| Review error logs | Check structured logs for `level=error` | Look for repeated errors, model failures, DB timeouts |
| Check stale jobs | Query `jobs` table for `status='processing'` older than 1 hour | `reap_stale_jobs` runs every 5 min via Celery Beat, but verify |

### Weekly

| Task | How |
|------|-----|
| Review rate limit hits | Check Redis keys `ratelimit:*` or logs for 429 responses |
| Check disk usage | Model files in `./models/` (~2GB total); DB size; Redis memory |
| Review webhook delivery failures | Check `deliver_webhook` task failures in Celery logs |
| Verify job cleanup | `cleanup_old_jobs` runs daily; confirm jobs older than `JOB_RETENTION_DAYS` (7) are deleted |

### Monthly

| Task | How |
|------|-----|
| Rotate API keys | Deactivate unused keys in `api_keys` table, call `invalidate_api_key_cache()` |
| Review WEBHOOK_SECRET_KEY | Rotate Fernet key if needed; re-encrypt existing secrets |
| Check dependency CVEs | Run `pip audit` or `safety check` against requirements |
| Review connection pool metrics | Check for pool exhaustion: async pool_size=20, sync pool_size=15 |

---

## 3. Monitoring

### Logs

Structured JSON logging via `structlog`. Key fields: `event`, `level`, `timestamp`, `request_id`.

**Log locations:**
- FastAPI: stdout (uvicorn)
- Celery worker: stdout (celery worker process)

**Critical log patterns to watch:**

| Pattern | Meaning | Action |
|---------|---------|--------|
| `"Database health check failed"` | PostgreSQL unreachable | Check DB connection, restart if needed |
| `"Inference timed out after Xs"` | ML model hung on an image | Check if recurring; may indicate bad model state |
| `"Worker: failed to load"` | Model didn't load on worker start | Check model files exist, disk space, memory |
| `"Redis not available"` | Redis down; rate limiting disabled | Restore Redis ASAP |
| `"Database engine was created in PID"` | Fork-safety violation | Ensure `--preload` is disabled for uvicorn/gunicorn |

### Metrics

Prometheus metrics at `/metrics` (requires API key in production).

**Key metrics:**

| Metric | What It Tells You |
|--------|-------------------|
| `http_requests_total` | Request volume by endpoint and status code |
| `http_request_duration_seconds` | Latency percentiles (p50, p95, p99) |
| `http_requests_in_progress` | Current concurrent requests |

**Alert thresholds (suggested):**

| Condition | Severity | Action |
|-----------|----------|--------|
| `/health` returns non-200 for > 30s | Critical | Restart service |
| `/health/ready` returns 503 | High | Check which component is down |
| p99 latency > 10s on `/faces/search` | Medium | Check DB performance, model state |
| Error rate > 5% on any endpoint | Medium | Check logs for root cause |
| Redis memory > 80% of limit | Medium | Check for key leaks, adjust maxmemory |

### Celery Monitoring

```bash
# Check active workers
celery -A src.workers.celery_app inspect active

# Check queues
celery -A src.workers.celery_app inspect active_queues

# Check reserved (prefetched) tasks
celery -A src.workers.celery_app inspect reserved
```

---

## 4. Performance Checks and Scaling

### Performance Baselines

| Endpoint | Expected Latency | Notes |
|----------|------------------|-------|
| `POST /blur/detect` | 50-200ms | Laplacian + FFT, CPU only |
| `POST /blur/classify` | 100-300ms | ONNX inference |
| `POST /faces/detect` | 200-800ms | InsightFace detection |
| `POST /faces/search` | 300-1500ms | Detection + embedding + DB search |
| `POST /faces/enroll` | 300-1000ms | Detection + embedding + DB write |
| `POST /bibs/recognize` | 500-2000ms | YOLO detection + PaddleOCR |

### Scaling Levers

| Bottleneck | Solution |
|------------|----------|
| **API throughput** | Increase `WORKERS` (uvicorn workers). Each worker holds all models in memory (~2GB). |
| **Batch processing** | Scale Celery workers. Use separate workers per queue: `celery -A src.workers.celery_app worker -Q blur -c 2` |
| **Face search latency** | Add pgvector HNSW index on `face_embeddings.embedding`. Current: exact search. |
| **Memory pressure** | Reduce `MAX_INFERENCE_DIMENSION` (default 2048). Lower = less memory per image, slightly lower accuracy. |
| **DB connections** | Async pool: `pool_size=20, max_overflow=10`. Sync pool: `pool_size=15, max_overflow=10`. Increase if pool exhaustion logged. |
| **Large event (many persons)** | Partition by `event_id`. Face search already filters by event_id when provided. |

### Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| API server | 2 CPU, 4GB RAM | 4 CPU, 8GB RAM |
| Celery worker | 2 CPU, 4GB RAM | 4 CPU, 8GB RAM |
| PostgreSQL | 1 CPU, 1GB RAM | 2 CPU, 4GB RAM |
| Redis | 0.5 CPU, 512MB RAM | 1 CPU, 1GB RAM |

---

## 5. Security Checks and Abuse Prevention

### Security Checklist

| Check | How to Verify |
|-------|---------------|
| `DEBUG=false` in production | Check `.env`; app refuses to start if `DEBUG=true` AND `ENVIRONMENT=production` |
| API docs disabled | `/docs` and `/redoc` return 404 when `DEBUG=false` |
| API key required on all endpoints | Test without `X-API-Key` header; expect 401 |
| Rate limiting active | Check Redis is up; test burst requests; expect 429 |
| WEBHOOK_SECRET_KEY set | Check `.env`; if unset, webhook secrets stored as plaintext (logged as warning on startup) |
| CORS origins restricted | Check `ALLOWED_ORIGINS` in `.env`; should NOT be `*` in production |
| Security headers present | Check response headers: `X-Content-Type-Options`, `X-Frame-Options`, HSTS |
| SSRF protection on webhooks | Webhook delivery blocks private IPs (10.x, 172.16.x, 192.168.x, 127.x) |

### Abuse Vectors and Mitigations

| Vector | Mitigation | Config |
|--------|------------|--------|
| **Image bombs** | Pillow decompression bomb limit (16M pixels), max file size, max dimension 4096px | `MAX_FILE_SIZE`, `MAX_DIMENSION` |
| **Batch flooding** | Max batch size, max active jobs per key, rate limiting | `MAX_BATCH_SIZE=20`, `MAX_ACTIVE_JOBS_PER_KEY=10` |
| **Inference DoS** | Per-image timeout (120s), request timeout (60s) | `INFERENCE_TIMEOUT`, `TimeoutMiddleware` |
| **Embedding spam** | Duplicate detection (same person + image hash), min enrollment confidence | `FACE_MIN_ENROLLMENT_CONFIDENCE=0.7` |
| **Webhook SSRF** | IP-literal check at registration, DNS resolution + private IP block at delivery | Built-in |
| **API key brute force** | Rate limiting, SHA-256 hashed storage | Rate tiers: free=60/min, pro=300/min |

### API Key Management

```sql
-- Create a new API key (do this, then give the raw key to the client)
-- The key_hash is SHA-256 of the raw key
INSERT INTO api_keys (id, key_hash, name, scopes, rate_tier, active)
VALUES (gen_random_uuid(), encode(sha256('raw-key-here'::bytea), 'hex'),
        'Client Name', '["*"]', 'free', true);

-- Deactivate a key
UPDATE api_keys SET active = false WHERE id = 'key-uuid';

-- After deactivating, invalidate the Redis cache (5-min window otherwise):
-- Call invalidate_api_key_cache(redis, 'raw-key-string') from Python
```

---

## 6. Model Lifecycle

### Model Inventory

| Model | Path | Format | Size | Required |
|-------|------|--------|------|----------|
| InsightFace buffalo_l | `models/buffalo_l/` | ONNX (multiple) | ~500MB | Yes |
| Blur classifier | `models/blur_classifier/blur_classifier.onnx` | ONNX | ~20MB | No (optional) |
| Blur class names | `models/blur_classifier/class_names.json` | JSON | <1KB | With classifier |
| Bib detector (YOLO) | `models/bib_detection/yolov8n_bib.onnx` | ONNX | ~12MB | No (optional) |
| PaddleOCR | Auto-downloaded by PaddleOCR | Multiple | ~150MB | Yes |

### Updating a Model

1. **Stop accepting new requests** (optional: set maintenance mode or drain)
2. **Place new model file** in the correct path under `models/`
3. **Restart the service** -- models are loaded at startup (`ModelRegistry.load_all()`) and at worker fork (`load_models_on_worker_start()`)
4. **Verify** via `/api/v1/health/ready` and test requests
5. **Keep the old model file** as a rollback copy

### Retraining the Blur Classifier

1. Train using YOLOv8 classification (see `docs/training-guide-bib.md` for reference)
2. Export to ONNX: `model.export(format='onnx')`
3. Ensure `class_names.json` matches the training class order (alphabetical by default)
4. The classifier validates class count against model output at load time -- mismatches disable the model
5. Place both files in `models/blur_classifier/` and restart

### Retraining the Bib Detector

1. Train YOLOv8n detection model on bib region annotations
2. Export to ONNX: `model.export(format='onnx')`
3. Place at `models/bib_detection/yolov8n_bib.onnx` and restart

### Rollback

1. Replace the model file with the previous version
2. Restart the service
3. No database migration needed -- models are stateless

---

## 7. Data Handling and Storage

### Database Tables

| Table | Purpose | Growth Pattern | Cleanup |
|-------|---------|----------------|---------|
| `persons` | Enrolled runner profiles | Grows with enrollments | Manual delete via API or GDPR erasure endpoint |
| `face_embeddings` | 512-dim vectors (pgvector) | Grows with enrollments (multiple per person) | Cascade-deleted with person |
| `jobs` | Batch job tracking | Grows with batch requests | Auto-cleaned after `JOB_RETENTION_DAYS` (7) by Celery Beat |
| `api_keys` | API key hashes + metadata | Slow growth | Manual management |
| `webhook_subscriptions` | Webhook registrations | Slow growth | Manual delete via API |

### Image Storage

Images are **NOT stored on disk or in the database**. The pipeline is:

1. Client uploads image via multipart/form-data
2. Image is decoded to numpy array in memory
3. ML inference runs on the array
4. Only the results (embeddings, bib numbers, blur scores) are stored
5. Raw image bytes are discarded after the request

For batch jobs, images are base64-encoded into the Celery message (stored temporarily in Redis). They are discarded after processing.

### Embedding Management

```sql
-- Count embeddings per person
SELECT p.name, COUNT(fe.id) as emb_count
FROM persons p
LEFT JOIN face_embeddings fe ON p.id = fe.person_id
GROUP BY p.id ORDER BY emb_count DESC;

-- Find persons with no embeddings (cleanup candidates)
SELECT p.id, p.name FROM persons p
LEFT JOIN face_embeddings fe ON p.id = fe.person_id
WHERE fe.id IS NULL;

-- Database size check
SELECT pg_size_pretty(pg_database_size('eventai'));

-- Embeddings table size
SELECT pg_size_pretty(pg_total_relation_size('face_embeddings'));
```

### GDPR / Data Erasure

```
DELETE /api/v1/faces/persons/{person_id}
```

This cascade-deletes the person and all their embeddings. The endpoint requires `faces:delete` scope.

---

## 8. Deployment and Update Process

### Initial Deployment

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env: set DATABASE_URL, REDIS_URL, WEBHOOK_SECRET_KEY, ALLOWED_ORIGINS

# 2. Start infrastructure
docker compose up -d db redis

# 3. Run database migrations
alembic upgrade head

# 4. Place model files in ./models/ directory

# 5. Start the API and worker
docker compose up -d ai-api celery-worker
```

### Updating the Application

```bash
# 1. Pull latest code
git pull origin main

# 2. Check for new migrations
alembic history --verbose  # compare with current: alembic current

# 3. Run migrations (if any)
alembic upgrade head

# 4. Rebuild and restart
docker compose build ai-api celery-worker
docker compose up -d ai-api celery-worker

# 5. Verify
curl http://localhost:8000/api/v1/health
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/api/v1/health/ready
```

### Rolling Back

```bash
# 1. Revert code
git checkout <previous-commit>

# 2. Downgrade migrations (if the update added any)
alembic downgrade -1   # one step back

# 3. Rebuild and restart
docker compose build ai-api celery-worker
docker compose up -d ai-api celery-worker
```

### Environment Variables Reference (Critical)

| Variable | Default | Must Set in Production |
|----------|---------|----------------------|
| `DEBUG` | `false` | Verify `false` |
| `ENVIRONMENT` | `development` | Set to `production` |
| `DATABASE_URL` | localhost | Set to production DB |
| `REDIS_URL` | localhost | Set to production Redis |
| `WEBHOOK_SECRET_KEY` | (empty) | Generate: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `ALLOWED_ORIGINS` | localhost:3000 | Set to actual frontend origin(s) |
| `WORKERS` | 2 | Scale based on CPU cores |

---

## 9. Failure Handling and Recovery

### Common Failures

| Symptom | Likely Cause | Recovery |
|---------|-------------|----------|
| `/health` returns non-200 | Process crashed | Check logs, restart: `docker compose restart ai-api` |
| `/health/ready` shows `database: false` | PostgreSQL down or unreachable | Check DB status, connection string, restart DB |
| `/health/ready` shows `redis: false` | Redis down | Restart Redis. API continues without cache/rate limiting |
| `/health/ready` shows `models_loaded: false` | Model files missing or corrupt | Check `./models/` directory, re-download models, restart |
| Jobs stuck in `processing` | Worker crashed mid-task | `reap_stale_jobs` auto-marks them failed after ~65 min. Or manually: `UPDATE jobs SET status='failed', error='manual reset' WHERE status='processing' AND created_at < NOW() - INTERVAL '1 hour'` |
| 503 on ML endpoints | Model not loaded | Check startup logs for model loading errors |
| 429 Too Many Requests | Rate limit hit | Normal behavior. Client should respect `Retry-After` header |
| 504 Gateway Timeout | Request exceeded 60s | Check if image is unusually large; check model performance |

### PostgreSQL Recovery

```bash
# Check if DB is accepting connections
docker compose exec db pg_isready

# Check active connections (pool exhaustion)
docker compose exec db psql -U postgres -d eventai -c "SELECT count(*) FROM pg_stat_activity WHERE datname='eventai';"

# Kill idle connections if pool is exhausted
docker compose exec db psql -U postgres -d eventai -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='eventai' AND state='idle' AND state_change < NOW() - INTERVAL '10 minutes';"
```

### Celery Worker Recovery

```bash
# Check if workers are alive
celery -A src.workers.celery_app inspect ping

# Restart workers (graceful -- finishes current tasks)
docker compose restart celery-worker

# Force restart (kills current tasks -- use only if stuck)
docker compose kill celery-worker && docker compose up -d celery-worker

# Purge all pending tasks (nuclear option -- loses queued work)
celery -A src.workers.celery_app purge
```

### Full System Recovery

```bash
# 1. Stop everything
docker compose down

# 2. Verify infrastructure
docker compose up -d db redis
# Wait for health checks to pass

# 3. Run any pending migrations
alembic upgrade head

# 4. Start application
docker compose up -d ai-api celery-worker

# 5. Verify all components
curl http://localhost:8000/api/v1/health
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/api/v1/health/ready
```

---

## 10. Testing and Validation

### Before Deploying Updates

```bash
# 1. Lint
ruff check src/ tests/

# 2. Type check
mypy src/

# 3. Unit tests
pytest tests/unit/ -v --tb=short

# 4. Syntax check all Python files
python -m py_compile src/main.py
# (or batch: find src -name "*.py" -exec python -m py_compile {} \;)
```

### After Deploying Updates

Run these checks against the live service:

```bash
API_KEY="your-api-key"
BASE="http://localhost:8000/api/v1"

# 1. Health checks
curl -s $BASE/health | python -m json.tool
curl -s -H "X-API-Key: $API_KEY" $BASE/health/ready | python -m json.tool

# 2. Blur detection (smoke test)
curl -s -X POST -H "X-API-Key: $API_KEY" \
  -F "file=@test_image.jpg" \
  $BASE/blur/detect | python -m json.tool

# 3. Face detection (smoke test)
curl -s -X POST -H "X-API-Key: $API_KEY" \
  -F "file=@test_image.jpg" \
  $BASE/faces/detect | python -m json.tool

# 4. Bib recognition (smoke test)
curl -s -X POST -H "X-API-Key: $API_KEY" \
  -F "file=@test_bib.jpg" \
  $BASE/bibs/recognize | python -m json.tool

# 5. Check security headers
curl -s -I -H "X-API-Key: $API_KEY" $BASE/health/ready | grep -i "x-content-type-options\|x-frame-options\|strict-transport"

# 6. Check rate limiting
curl -s -I -H "X-API-Key: $API_KEY" $BASE/health | grep -i "x-ratelimit"
```

### Validation Criteria

| Check | Pass Condition |
|-------|---------------|
| Health liveness | HTTP 200, `status: "alive"` |
| Health readiness | HTTP 200, all three checks true |
| Blur detect | HTTP 200, `success: true`, `is_blurry` field present |
| Face detect | HTTP 200, `success: true`, `faces_detected >= 0` |
| Bib recognize | HTTP 200, `success: true`, `bibs_detected >= 0` |
| Security headers | `x-content-type-options: nosniff` present |
| Rate limit headers | `X-RateLimit-Remaining` present |
| Error on no API key | HTTP 401 when `X-API-Key` header omitted |

---

## Quick Reference

### Ports

| Service | Port |
|---------|------|
| FastAPI (uvicorn) | 8000 |
| PostgreSQL | 5432 |
| Redis | 6379 |

### Task Queues

| Queue | Tasks |
|-------|-------|
| `default` | Webhooks, maintenance |
| `blur` | `blur.detect_batch`, `blur.classify_batch` |
| `face` | `faces.process_batch`, `faces.enroll_batch` |
| `bib` | `bibs.recognize_batch` |

### Celery Beat Schedule

| Task | Interval | Purpose |
|------|----------|---------|
| `maintenance.reap_stale_jobs` | 5 minutes | Mark stuck jobs as failed |
| `maintenance.cleanup_old_jobs` | 24 hours | Delete jobs older than 7 days |

### Rate Limit Tiers

| Tier | Requests/min | Burst |
|------|-------------|-------|
| `free` | 60 | 60 |
| `pro` | 300 | 300 |
| `internal` | 1000 | 1000 |
