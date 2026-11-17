-- PE-1 (photo-to-event relevance): the scored snapshot of each batch and its looked-up photos.
-- A batch is one event_photographer row (one photographer in one event). Written only by
-- service/relevance. Nothing here hides, holds or deletes anything.
--
-- The content-hash index takes a short write lock on photos while it builds (no CONCURRENTLY
-- inside a Flyway transaction): merge and release off race day.

CREATE TABLE batch_relevance (
    id                 UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id           UUID         NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    photographer_id    UUID         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_swept_at      TIMESTAMPTZ,
    scored_at          TIMESTAMPTZ,
    scored_upload_at   TIMESTAMPTZ,
    scored_event_date  DATE,
    final_scored_at    TIMESTAMPTZ,
    photos_live        INTEGER,
    photos_processing  INTEGER,
    lookups            INTEGER,
    tight_hits         INTEGER,
    weak_hits          INTEGER,
    exact_matches      INTEGER,
    xmp_matches        INTEGER,
    dated_photos       INTEGER,
    dated_outside      INTEGER,
    metadata_pending   INTEGER,
    indexed_photos     INTEGER,
    bibbed_photos      INTEGER,
    bib_overlap_photos INTEGER,
    copy_low           BOOLEAN,
    date_low           BOOLEAN,
    level              VARCHAR(8)   CHECK (level IN ('LOW')),
    CONSTRAINT uq_batch_relevance_batch UNIQUE (event_id, photographer_id)
);

CREATE INDEX idx_batch_relevance_photographer ON batch_relevance (photographer_id);

-- One row per looked-up photo: the row is the "already looked up" mark. Match ids carry no
-- foreign key (like flags.target_id), so deleting the matched photo later keeps the distance.
CREATE TABLE batch_relevance_photos (
    photo_id        UUID         PRIMARY KEY REFERENCES photos (id) ON DELETE CASCADE,
    batch_id        UUID         NOT NULL REFERENCES batch_relevance (id) ON DELETE CASCADE,
    looked_up_at    TIMESTAMPTZ  NOT NULL,
    match_photo_id  UUID,
    match_distance  SMALLINT     CHECK (match_distance BETWEEN 0 AND 64),
    match_pair      VARCHAR(6)   CHECK (match_pair IN ('CLEAN', 'CENTRE', 'MARKED')),
    exact_photo_id  UUID
);

CREATE INDEX idx_batch_relevance_photos_batch ON batch_relevance_photos (batch_id);

-- The exact-copy look-up across all events (C17). The V24 index leads with photographer_id.
CREATE INDEX idx_photos_content_hash ON photos (content_hash) WHERE content_hash IS NOT NULL;

-- The QuickPitik photo id read from an upload's XMP (PE-1.4). A column, not a metadata field,
-- so the public metadata response never carries it.
ALTER TABLE photo_metadata ADD COLUMN xmp_photo_id UUID;
