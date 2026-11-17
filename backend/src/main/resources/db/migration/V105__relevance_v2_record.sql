-- RV-1.1 (Photo Relevance v2, record-only): every table record-only needs, in one file (plan 5.4).
-- Every statement is a new table or a nullable column without a default, so nothing takes a long
-- lock or rewrites a table. Nothing reads these until later RV tasks; RELEVANCE_MODE decides.
-- Written only by service/relevance.

-- Words read off the batch's photos, as counts per batch (locked step 1, G14). No photo ids.
CREATE TABLE batch_relevance_words (
    batch_id UUID        NOT NULL REFERENCES batch_relevance (id) ON DELETE CASCADE,
    word     VARCHAR(40) NOT NULL,
    photos   INTEGER     NOT NULL CHECK (photos > 0),
    PRIMARY KEY (batch_id, word)
);

-- One row per photo: the "words counted" mark (R5) and the measurement, stored once (S1).
CREATE TABLE relevance_photo_checks (
    photo_id         UUID PRIMARY KEY REFERENCES photos (id) ON DELETE CASCADE,
    batch_id         UUID NOT NULL REFERENCES batch_relevance (id) ON DELETE CASCADE,
    words_counted_at TIMESTAMPTZ,
    measured_at      TIMESTAMPTZ,
    fg_sharpness     DOUBLE PRECISION,
    fg_brightness    DOUBLE PRECISION,
    sharpness        DOUBLE PRECISION,
    brightness       DOUBLE PRECISION,
    person_conf      DOUBLE PRECISION,
    race_conf        DOUBLE PRECISION,
    measure_tries    SMALLINT NOT NULL DEFAULT 0,
    measure_error    VARCHAR(40)
);
CREATE INDEX idx_relevance_photo_checks_batch ON relevance_photo_checks (batch_id);

ALTER TABLE batch_relevance
    ADD COLUMN check_claim       UUID,
    ADD COLUMN check_claim_until TIMESTAMPTZ,
    ADD COLUMN check_served_at   TIMESTAMPTZ,
    ADD COLUMN checked_upload_at TIMESTAMPTZ,
    ADD COLUMN checked_photos    INTEGER,
    ADD COLUMN words_purged_at   TIMESTAMPTZ;

-- One row per check of a batch.
CREATE TABLE relevance_cases (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id        UUID        NOT NULL REFERENCES batch_relevance (id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    trigger         VARCHAR(8)  NOT NULL CHECK (trigger IN ('QUIET', 'COUNT', 'RULES', 'SEEDED')),
    upload_at       TIMESTAMPTZ NOT NULL,
    batch_photos    INTEGER     NOT NULL,
    stage           VARCHAR(10) NOT NULL CHECK (stage IN ('SAMPLING', 'DECIDING', 'DECIDED', 'APPLIED', 'CLOSED')),
    sample_round    SMALLINT    NOT NULL DEFAULT 1,
    sample_size     INTEGER,
    measured        INTEGER,
    blurry          INTEGER,
    dark            INTEGER,
    not_race        INTEGER,
    junk            INTEGER,
    words_photos    INTEGER,
    band            VARCHAR(8)  CHECK (band IN ('UNDER', 'MIXED', 'OVER', 'SMALL', 'UNKNOWN')),
    copy_low        BOOLEAN,
    date_low        BOOLEAN,
    decide_attempts SMALLINT    NOT NULL DEFAULT 0,
    model_verdict   VARCHAR(12),
    verdict         VARCHAR(12) CHECK (verdict IN ('FITS', 'MIXED', 'JUNK_DUMP', 'WRONG_EVENT', 'COPIED', 'CANT_TELL')),
    verdict_note    VARCHAR(40),
    admin_reason    TEXT,
    evidence_used   TEXT,
    summary_sent    TEXT,
    model           VARCHAR(60),
    prompt_version  VARCHAR(12),
    decided_at      TIMESTAMPTZ,
    applied_mode    VARCHAR(12) CHECK (applied_mode IN ('RECORD_ONLY', 'AUTO_HIDE')),
    action          VARCHAR(10) CHECK (action IN ('NONE', 'HIDE_JUNK', 'HIDE_ALL')),
    applied_at      TIMESTAMPTZ,
    closed_at       TIMESTAMPTZ,
    close_reason    VARCHAR(24)
);
CREATE INDEX idx_relevance_cases_batch ON relevance_cases (batch_id, created_at DESC);
-- One check in flight per batch, and one current case per batch.
CREATE UNIQUE INDEX uq_relevance_cases_in_flight ON relevance_cases (batch_id)
    WHERE stage IN ('SAMPLING', 'DECIDING', 'DECIDED');
CREATE UNIQUE INDEX uq_relevance_cases_current ON relevance_cases (batch_id) WHERE stage = 'APPLIED';

-- The photos a check sampled (so the sample is fixed across ticks and admins can see it).
CREATE TABLE relevance_case_photos (
    case_id  UUID     NOT NULL REFERENCES relevance_cases (id) ON DELETE CASCADE,
    photo_id UUID     NOT NULL REFERENCES photos (id) ON DELETE CASCADE,
    round    SMALLINT NOT NULL,
    PRIMARY KEY (case_id, photo_id)
);

-- Stored admin alerts (R12). The live ping only refreshes the page.
CREATE TABLE relevance_alerts (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id   UUID         NOT NULL REFERENCES batch_relevance (id) ON DELETE CASCADE,
    case_id    UUID         REFERENCES relevance_cases (id) ON DELETE SET NULL,
    kind       VARCHAR(12)  NOT NULL CHECK (kind IN ('BIG_HIDE', 'COPIED', 'APPEAL', 'THIRD_CASE')),
    dedupe_key VARCHAR(120) NOT NULL UNIQUE,
    would_have BOOLEAN      NOT NULL,
    photos     INTEGER,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    seen_at    TIMESTAMPTZ,
    seen_by    UUID         REFERENCES users (id) ON DELETE SET NULL
);
CREATE INDEX idx_relevance_alerts_unseen ON relevance_alerts (created_at DESC) WHERE seen_at IS NULL;
