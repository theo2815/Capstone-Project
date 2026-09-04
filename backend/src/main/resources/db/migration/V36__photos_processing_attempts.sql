-- V36: async-watermark retry budget (mirrors indexing_attempts). Photos are now
-- created status=PROCESSING and flipped LIVE by PhotoWatermarkTrigger once the
-- watermarked derivative lands in storage; this column bounds how many times a
-- semantically-failing photo (undecodable bytes) is re-attempted before the
-- sweep stops re-driving it. Transport failures do not consume the budget.
ALTER TABLE photos
    ADD COLUMN processing_attempts INT NOT NULL DEFAULT 0;

-- The reconcile sweep scans for stuck PROCESSING rows every minute; steady-state
-- that set is empty-to-tiny, so a partial index keeps the sweep off the main
-- photos heap entirely.
CREATE INDEX idx_photos_watermark_backlog
    ON photos (uploaded_at)
    WHERE status = 'PROCESSING';
