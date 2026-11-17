-- PE-2.1: all of PE-2's schema (photo-to-event relevance, flag mode) in one file, so PE-2.2 to
-- PE-2.4 need none. Nothing here changes behaviour on its own: RELEVANCE_MODE decides.

-- PE-2.2's payout hold: an admin's confirmation on a relevance flag (plan 5.10). resolved_by and
-- resolved_at cannot carry it: hide already stamps them.
ALTER TABLE flags
    ADD COLUMN payout_hold_at TIMESTAMPTZ,
    ADD COLUMN payout_hold_by UUID REFERENCES users (id) ON DELETE SET NULL;

-- The one open relevance flag a Low batch raised or extended (plan 5.9).
ALTER TABLE batch_relevance
    ADD COLUMN flag_id UUID REFERENCES flags (id) ON DELETE SET NULL,
    ADD COLUMN flagged_at TIMESTAMPTZ;
CREATE INDEX idx_batch_relevance_flag ON batch_relevance (flag_id) WHERE flag_id IS NOT NULL;

-- PE-2.4's whole-batch hide: which photos a relevance flag hid, so one restore rule can tell when a
-- photo may come back (plan 5.12). Written only by service/admin.
CREATE TABLE flag_hidden_photos (
    flag_id     UUID        NOT NULL REFERENCES flags (id) ON DELETE CASCADE,
    photo_id    UUID        NOT NULL REFERENCES photos (id) ON DELETE CASCADE,
    hidden_at   TIMESTAMPTZ NOT NULL,
    restored_at TIMESTAMPTZ,
    PRIMARY KEY (flag_id, photo_id)
);
CREATE INDEX idx_flag_hidden_photos_photo ON flag_hidden_photos (photo_id);
