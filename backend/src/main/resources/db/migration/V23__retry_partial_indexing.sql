-- V23: let the reconciliation sweep retry PARTIAL photos
--
-- A photo lands in PARTIAL when exactly one of face/bib hit a *transient* ai-api
-- error while the other succeeded. The V21 reconcile index only covered
-- PENDING/FAILED, so a PARTIAL photo's failed half was stranded forever — a brief
-- ai-api blip silently dropping a bib (or a face), against the pipeline's
-- self-healing goal. The per-photo sweep now treats PARTIAL as retryable
-- (PhotoIndexingTrigger.RETRYABLE_STATUSES); widen the partial index to match so
-- the probe stays index-backed. index() is idempotent, so re-driving a PARTIAL
-- photo cannot duplicate embeddings.
--
-- The batch-drain backlog query still selects only PENDING/FAILED; since that set
-- is a subset of this index's predicate, the index continues to serve it too.

DROP INDEX IF EXISTS idx_photos_indexing_pending;

CREATE INDEX idx_photos_indexing_pending
    ON photos (indexing_status)
    WHERE indexing_status IN ('PENDING', 'FAILED', 'PARTIAL');
