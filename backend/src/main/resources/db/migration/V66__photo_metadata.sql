ALTER TABLE photographer_settings
    ADD COLUMN metadata_public_by_default BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE event_photographer
    ADD COLUMN metadata_public BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE photo_upload_intents
    ADD COLUMN original_filename VARCHAR(255);

CREATE TABLE photo_metadata (
    photo_id UUID PRIMARY KEY REFERENCES photos(id) ON DELETE CASCADE,
    original_filename VARCHAR(255),
    original_size_bytes BIGINT CHECK (original_size_bytes > 0),
    fields JSONB,
    state VARCHAR(16) NOT NULL DEFAULT 'PENDING'
        CHECK (state IN ('PENDING', 'PROCESSING', 'READY', 'EMPTY', 'FAILED', 'UNAVAILABLE')),
    extractor_version INTEGER NOT NULL DEFAULT 1 CHECK (extractor_version > 0),
    extracted_at TIMESTAMPTZ,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TIMESTAMPTZ,
    lease_token UUID,
    lease_until TIMESTAMPTZ
);

CREATE INDEX idx_photo_metadata_due ON photo_metadata (next_attempt_at, photo_id)
    WHERE state IN ('PENDING', 'PROCESSING', 'FAILED');
