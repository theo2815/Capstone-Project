-- V21: async photo indexing status
--
-- Face enroll + bib OCR (the ai-api "indexing" of an uploaded photo) moved off
-- the synchronous upload request into a background worker (PhotoIndexingService).
-- These columns make that worker durable and observable: a reconciliation sweep
-- re-drives PENDING/FAILED rows, and indexed_at / indexing_error record outcomes.
--
-- Existing rows default to INDEXED: they were uploaded under the old synchronous
-- path, so their faces/bibs (if any) are already written and must not be
-- re-processed by the sweep. New uploads set the value explicitly via JPA
-- (PENDING when ai-api is enabled, SKIPPED when disabled).

ALTER TABLE photos
    ADD COLUMN indexing_status   VARCHAR(16)  NOT NULL DEFAULT 'INDEXED'
        CHECK (indexing_status IN ('PENDING', 'INDEXED', 'PARTIAL', 'FAILED', 'SKIPPED')),
    ADD COLUMN indexed_at        TIMESTAMPTZ,
    ADD COLUMN indexing_attempts INTEGER      NOT NULL DEFAULT 0,
    ADD COLUMN indexing_error    TEXT;

-- The reconciliation sweep only ever scans PENDING/FAILED rows; a partial index
-- keeps that probe cheap as the photos table grows into the millions.
CREATE INDEX idx_photos_indexing_pending
    ON photos (indexing_status)
    WHERE indexing_status IN ('PENDING', 'FAILED');
