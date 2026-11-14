-- RV-2.1: Photo Relevance v2, migration 2 of 2 (part 2, A.1): everything acting needs. Nothing here changes
-- behaviour on its own: RELEVANCE_MODE decides, and no code writes the new rows yet.

-- Holds: flag_hidden_photos becomes the one hold table (part 1, 5.10). A row belongs to a flag (held_by FLAG,
-- version 1, written by service/admin) or to a batch (AUTO, ADMIN, PHOTOGRAPHER, written by service/relevance),
-- never both. Version 1's ON CONFLICT (flag_id, photo_id) keeps working on the new unique constraint.
-- The new id column has a volatile default, so this rewrites the table under an exclusive lock.
ALTER TABLE flag_hidden_photos DROP CONSTRAINT flag_hidden_photos_pkey;
ALTER TABLE flag_hidden_photos
    ADD COLUMN id          UUID NOT NULL DEFAULT gen_random_uuid(),
    ALTER COLUMN flag_id DROP NOT NULL,
    ADD COLUMN batch_id    UUID REFERENCES batch_relevance (id) ON DELETE CASCADE,
    ADD COLUMN case_id     UUID REFERENCES relevance_cases (id) ON DELETE SET NULL,
    ADD COLUMN held_by     VARCHAR(12) NOT NULL DEFAULT 'FLAG'
                           CHECK (held_by IN ('FLAG', 'AUTO', 'ADMIN', 'PHOTOGRAPHER')),
    ADD COLUMN restored_by VARCHAR(12) CHECK (restored_by IN ('AUTO', 'ADMIN', 'PHOTOGRAPHER')),
    ADD CONSTRAINT flag_hidden_photos_pkey PRIMARY KEY (id),
    ADD CONSTRAINT uq_flag_hidden_photos_flag UNIQUE (flag_id, photo_id),
    ADD CONSTRAINT uq_flag_hidden_photos_batch UNIQUE (batch_id, photo_id, held_by),
    ADD CONSTRAINT ck_flag_hidden_photos_owner CHECK (
        (flag_id IS NOT NULL AND batch_id IS NULL AND held_by = 'FLAG')
        OR (flag_id IS NULL AND batch_id IS NOT NULL AND held_by <> 'FLAG')),
    ADD CONSTRAINT ck_flag_hidden_photos_restore CHECK (
        batch_id IS NULL OR (restored_at IS NULL) = (restored_by IS NULL));
CREATE INDEX idx_flag_hidden_photos_batch ON flag_hidden_photos (batch_id) WHERE batch_id IS NOT NULL;

-- The automatic actor (G17): a row has an admin or is automatic.
ALTER TABLE admin_decision_log
    ALTER COLUMN admin_id DROP NOT NULL,
    ADD COLUMN automatic BOOLEAN NOT NULL DEFAULT false,
    ADD CONSTRAINT ck_admin_decision_log_actor CHECK (automatic OR admin_id IS NOT NULL);

-- What acting needs on a case.
ALTER TABLE relevance_cases
    ADD COLUMN hidden_photos    INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN hide_finished_at TIMESTAMPTZ,
    ADD COLUMN scan_state       VARCHAR(8) NOT NULL DEFAULT 'NONE'
                                CHECK (scan_state IN ('NONE', 'QUEUED', 'RUNNING', 'FINISHED')),
    ADD COLUMN scan_served_at   TIMESTAMPTZ,
    ADD COLUMN scan_finished_at TIMESTAMPTZ,
    ADD COLUMN scan_measured    INTEGER,
    ADD COLUMN scan_junk        INTEGER,
    ADD COLUMN scan_blurry      INTEGER,
    ADD COLUMN flipped_at       TIMESTAMPTZ,
    ADD COLUMN notice_at        TIMESTAMPTZ,
    ADD COLUMN confirmed_at     TIMESTAMPTZ,
    ADD COLUMN closed_by        UUID REFERENCES users (id) ON DELETE SET NULL,
    ADD COLUMN hold_flag_id     UUID REFERENCES flags (id) ON DELETE SET NULL;

-- The photographer's note (P7): V73's 21 kinds, copied exactly, and relevance_notice.
ALTER TABLE photographer_messages DROP CONSTRAINT photographer_messages_kind_check;
ALTER TABLE photographer_messages ADD CONSTRAINT photographer_messages_kind_check CHECK (kind IN (
    'verification_approved', 'verification_rejected', 'verification_reset', 'suspended', 'unsuspended',
    'force_edit', 'dispute_resolved', 'dispute_denied', 'dispute_escalated', 'payout_approved', 'payout_held',
    'payout_paid', 'payout_report_acknowledged', 'payout_report_resolved', 'admin_message', 'event_approved',
    'event_rejected', 'event_change_approved', 'event_change_rejected', 'photo_removal_request', 'photo_removal_outcome',
    'relevance_notice'
));
ALTER TABLE photographer_messages
    ADD COLUMN relevance_case_id UUID REFERENCES relevance_cases (id) ON DELETE SET NULL;

-- Appeals (G11): one per case, one open per batch.
CREATE TABLE relevance_appeals (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id            UUID        NOT NULL UNIQUE REFERENCES relevance_cases (id) ON DELETE CASCADE,
    batch_id           UUID        NOT NULL REFERENCES batch_relevance (id) ON DELETE CASCADE,
    photographer_id    UUID        NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    reason             TEXT        NOT NULL CHECK (char_length(reason) BETWEEN 1 AND 1000),
    status             VARCHAR(8)  NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'WON', 'LOST')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    evidence_opened_at TIMESTAMPTZ,
    evidence_opened_by UUID REFERENCES users (id) ON DELETE SET NULL,
    decided_at         TIMESTAMPTZ,
    decided_by         UUID REFERENCES users (id) ON DELETE SET NULL,
    decision_note      TEXT
);
CREATE UNIQUE INDEX uq_relevance_appeals_open ON relevance_appeals (batch_id) WHERE status = 'OPEN';

-- Confirmed cases that count (P6). No foreign key to the case or the event, so a count survives their deletion.
CREATE TABLE relevance_strikes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    photographer_id UUID        NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    case_id         UUID        NOT NULL UNIQUE,
    event_id        UUID        NOT NULL,
    verdict         VARCHAR(12) NOT NULL,
    confirmed_at    TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_relevance_strikes_photographer ON relevance_strikes (photographer_id, confirmed_at DESC);
