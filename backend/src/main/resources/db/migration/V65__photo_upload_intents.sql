CREATE TABLE photo_upload_intents (
    photo_id UUID PRIMARY KEY,
    photographer_id UUID NOT NULL REFERENCES users(id),
    event_id UUID NOT NULL REFERENCES events(id),
    request_id UUID,
    content_hash VARCHAR(64) NOT NULL
        CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    content_type VARCHAR(32) NOT NULL
        CHECK (content_type IN ('image/jpeg', 'image/png', 'image/webp')),
    size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
    staging_key VARCHAR(512) NOT NULL UNIQUE,
    original_key VARCHAR(512) NOT NULL UNIQUE,
    state VARCHAR(16) NOT NULL
        CHECK (state IN ('PENDING', 'VERIFYING', 'COMMITTED', 'REJECTED', 'EXPIRED')),
    lease_token UUID,
    lease_until TIMESTAMPTZ,
    result_photo_id UUID REFERENCES photos(id) ON DELETE SET NULL,
    error_code VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL,
    put_expires_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    cleanup_after TIMESTAMPTZ,
    cleanup_checked_at TIMESTAMPTZ,
    cleaned_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX uq_photo_upload_request
    ON photo_upload_intents (photographer_id, request_id)
    WHERE request_id IS NOT NULL;

CREATE INDEX idx_photo_upload_pending_owner
    ON photo_upload_intents (photographer_id, expires_at)
    WHERE state IN ('PENDING', 'VERIFYING');

CREATE INDEX idx_photo_upload_cleanup
    ON photo_upload_intents (cleanup_after, photo_id)
    WHERE cleanup_after IS NOT NULL;
