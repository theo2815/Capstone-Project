-- V15 — Surface the photographer_messages inbox to the photographer.
--
-- V10 created the table + writers (every admin action calls
-- AdminDecisionLogService.pushMessage). PR 10 deferred the read side:
-- "/me/photographer/inbox (path TBD, deferred from PR 7 since the FE
-- didn't surface it yet)". This migration adds the three columns the FE
-- needs to render parity with the prior FE-only mock + drops the
-- (now-stale) "no admin_message in CHECK constraint" gap:
--
--   admin_message kind     — admin → photographer free-form DM
--                            (POST /admin/users/{id}/message)
--   title TEXT NULL        — used by admin_message (subject); other kinds
--                            keep title NULL and the FE derives the label
--                            from MESSAGE_KIND_LABEL[kind]
--   removed_at TIMESTAMPTZ — soft-delete for the photographer's "Remove"
--                            action; GET /me/photographer/messages filters
--                            removed_at IS NULL
--
-- Indexes are recreated with `WHERE removed_at IS NULL` so the hot path
-- skips removed rows without rewriting the planner stats.

ALTER TABLE photographer_messages
    DROP CONSTRAINT photographer_messages_kind_check;

ALTER TABLE photographer_messages
    ADD CONSTRAINT photographer_messages_kind_check CHECK (
        kind IN (
            'verification_approved',
            'verification_rejected',
            'verification_reset',
            'suspended',
            'unsuspended',
            'force_edit',
            'dispute_resolved',
            'dispute_denied',
            'dispute_escalated',
            'payout_approved',
            'payout_held',
            'payout_paid',
            'payout_report_acknowledged',
            'payout_report_resolved',
            'admin_message'
        )
    );

ALTER TABLE photographer_messages
    ADD COLUMN title      TEXT,
    ADD COLUMN removed_at TIMESTAMPTZ;

DROP INDEX IF EXISTS idx_photographer_messages_photographer;
DROP INDEX IF EXISTS idx_photographer_messages_unread;

CREATE INDEX idx_photographer_messages_photographer
    ON photographer_messages (photographer_id, created_at DESC)
    WHERE removed_at IS NULL;

CREATE INDEX idx_photographer_messages_unread
    ON photographer_messages (photographer_id, created_at DESC)
    WHERE read_at IS NULL AND removed_at IS NULL;
