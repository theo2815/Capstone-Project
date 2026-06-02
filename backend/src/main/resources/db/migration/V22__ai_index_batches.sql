-- V22: bulk photo indexing (Phase C) — batch drain + job tracking + BATCHING state
--
-- Opt-in via app.ai-api.indexing-mode=batch. When on, the per-photo hot path
-- stands down and a scheduled drain groups PENDING photos per event into one
-- face-mega + one bib-mega job, then ingests results via webhook (prod) or poll
-- (dev/backstop). Inert until the mode flag is flipped — additive only.

-- Widen the V21 status CHECK to allow the in-flight BATCHING state. A photo
-- flipped to BATCHING has been submitted to ai-api in a mega job; the Phase-B
-- partial index (idx_photos_indexing_pending) covers only PENDING/FAILED, so
-- BATCHING rows are naturally excluded from the per-photo sweep. The batch
-- ingest settles them to INDEXED/PARTIAL/FAILED (or reverts to PENDING).
ALTER TABLE photos DROP CONSTRAINT photos_indexing_status_check;
ALTER TABLE photos ADD CONSTRAINT photos_indexing_status_check
    CHECK (indexing_status IN ('PENDING', 'BATCHING', 'INDEXED', 'PARTIAL', 'FAILED', 'SKIPPED'));

-- One batch = one event's slice of PENDING photos, submitted as a FACE mega job
-- and a BIB mega job. photo_ids are ordered so result index i maps to photo i
-- (ai-api also echoes a per-image ref = the photo id). A photo settles only when
-- BOTH its jobs are terminal.
CREATE TABLE ai_index_batches (
    id           UUID PRIMARY KEY,
    event_id     UUID NOT NULL,
    status       VARCHAR(16) NOT NULL,        -- SUBMITTED | FINALIZED | FAILED
    created_at   TIMESTAMPTZ NOT NULL,
    finalized_at TIMESTAMPTZ
);

-- Ordered photo membership of a batch (@ElementCollection @OrderColumn).
CREATE TABLE ai_index_batch_photos (
    batch_id UUID    NOT NULL REFERENCES ai_index_batches(id) ON DELETE CASCADE,
    idx      INTEGER NOT NULL,
    photo_id UUID    NOT NULL,
    PRIMARY KEY (batch_id, idx)
);

CREATE TABLE ai_index_jobs (
    id           UUID PRIMARY KEY,
    batch_id     UUID NOT NULL REFERENCES ai_index_batches(id) ON DELETE CASCADE,
    kind         VARCHAR(8)  NOT NULL,        -- FACE | BIB
    ai_job_id    VARCHAR(64) NOT NULL UNIQUE, -- idempotency anchor for receiver + poll
    status       VARCHAR(16) NOT NULL,        -- SUBMITTED | COMPLETED | FAILED
    submitted_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    error        TEXT
);

-- The poll/reconcile backstop only scans jobs still awaiting results.
CREATE INDEX idx_ai_index_jobs_open ON ai_index_jobs (status) WHERE status = 'SUBMITTED';
