-- Freeze gallery pagination against the moment a photo becomes runner-visible.
-- uploaded_at cannot do this because watermarking publishes asynchronously.
ALTER TABLE photos ADD COLUMN published_at TIMESTAMPTZ;

UPDATE photos
SET published_at = uploaded_at
WHERE status IN ('LIVE', 'HIDDEN');

-- Supports the per-photographer window used by the unfiltered event gallery.
CREATE INDEX idx_photos_event_live_photographer_rank
    ON photos (event_id, photographer_id, captured_at DESC NULLS LAST, uploaded_at DESC, id ASC)
    WHERE status = 'LIVE';
