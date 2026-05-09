-- V3: photos catalog + bib + face-person tags
-- photographer_id is NULL until PR 7 (photographer coverage) starts populating
-- real uploads. Until then this table stays empty in dev.

CREATE TABLE photos (
    id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id            UUID         NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    photographer_id     UUID                  REFERENCES users  (id) ON DELETE SET NULL,
    s3_key              VARCHAR(512) NOT NULL,
    thumbnail_s3_key    VARCHAR(512),
    watermark_s3_key    VARCHAR(512),
    blur_score          NUMERIC(10, 4),
    span                VARCHAR(20)  NOT NULL DEFAULT 'default'
                            CHECK (span IN ('default', 'wide')),
    tone                INTEGER      NOT NULL DEFAULT 0,
    km                  NUMERIC(5, 2),
    captured_at         TIMESTAMPTZ,
    uploaded_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    status              VARCHAR(20)  NOT NULL DEFAULT 'LIVE'
                            CHECK (status IN ('PROCESSING', 'LIVE', 'HIDDEN')),
    price_php           NUMERIC(12, 2) NOT NULL,
    alt_text            TEXT
);

CREATE INDEX idx_photos_event_captured_at ON photos (event_id, captured_at DESC, id);
CREATE INDEX idx_photos_event_uploaded_at ON photos (event_id, uploaded_at DESC, id);
CREATE INDEX idx_photos_event_status      ON photos (event_id, status);
CREATE INDEX idx_photos_photographer      ON photos (photographer_id);

CREATE TABLE photo_bibs (
    photo_id        UUID         NOT NULL REFERENCES photos (id) ON DELETE CASCADE,
    bib_number      VARCHAR(20)  NOT NULL,
    ocr_confidence  NUMERIC(5, 4) NOT NULL,
    PRIMARY KEY (photo_id, bib_number)
);

CREATE INDEX idx_photo_bibs_bib_number ON photo_bibs (bib_number);

CREATE TABLE photo_face_persons (
    photo_id     UUID         NOT NULL REFERENCES photos (id) ON DELETE CASCADE,
    face_index   INTEGER      NOT NULL,
    ai_person_id VARCHAR(100) NOT NULL,
    PRIMARY KEY (photo_id, face_index)
);

CREATE INDEX idx_photo_face_persons_ai_person_id ON photo_face_persons (ai_person_id);
