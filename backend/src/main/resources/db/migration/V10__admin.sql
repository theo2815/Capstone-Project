-- V10: Phase G admin (PR 10).
--   users.suspended_at + suspension_reason
--                  : suspension applies to RUNNER + PHOTOGRAPHER alike, so it
--                    lives on users (not photographer_settings). NULL =
--                    not suspended; non-NULL = suspended at that instant
--                    with the matching reason snapshot.
--   admin_decision_log
--                  : append-only audit trail. group_id is nullable per Q-A4 —
--                    bulk admin actions (e.g. POST /admin/payouts/bulk) share
--                    one UUID so the dashboard can collapse the kpi count by
--                    `group_id IS NOT NULL`. Target columns are nullable
--                    because each decision targets exactly one of user /
--                    payout / dispute (sometimes none, e.g. event-edit logs
--                    target_event_id).
--   photographer_messages
--                  : DB-backed inbox per Q-A2. Admin actions write a row in
--                    the same TX as the decision-log entry; the photographer
--                    surface polls for unread rows. No external webhook /
--                    email integration in v1.
--   flags          : minimal scaffold for the ADMIN_FLAGS_ENABLED-gated
--                    flagging surface. Body schema kept narrow — fuller
--                    triage workflow lands when the queue actually starts
--                    receiving rows (Q-A5 follow-up).

ALTER TABLE users
    ADD COLUMN suspended_at TIMESTAMPTZ,
    ADD COLUMN suspension_reason VARCHAR(500);

CREATE INDEX idx_users_suspended_at ON users (suspended_at) WHERE suspended_at IS NOT NULL;


CREATE TABLE admin_decision_log (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    admin_id            UUID            NOT NULL REFERENCES users (id) ON DELETE SET NULL,
    target_user_id      UUID                       REFERENCES users (id) ON DELETE SET NULL,
    target_payout_id    VARCHAR(80)                REFERENCES payout_cycles (id) ON DELETE SET NULL,
    target_dispute_id   UUID                       REFERENCES disputes (id) ON DELETE SET NULL,
    target_event_id     UUID                       REFERENCES events (id) ON DELETE SET NULL,
    decision            VARCHAR(40)     NOT NULL,
    reason              TEXT,
    meta                JSONB,
    group_id            UUID,
    decided_at          TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX idx_admin_decision_log_decided_at        ON admin_decision_log (decided_at DESC);
CREATE INDEX idx_admin_decision_log_target_user       ON admin_decision_log (target_user_id, decided_at DESC) WHERE target_user_id IS NOT NULL;
CREATE INDEX idx_admin_decision_log_target_payout     ON admin_decision_log (target_payout_id, decided_at DESC) WHERE target_payout_id IS NOT NULL;
CREATE INDEX idx_admin_decision_log_target_dispute    ON admin_decision_log (target_dispute_id, decided_at DESC) WHERE target_dispute_id IS NOT NULL;
CREATE INDEX idx_admin_decision_log_group             ON admin_decision_log (group_id) WHERE group_id IS NOT NULL;


CREATE TABLE photographer_messages (
    id                   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    photographer_id      UUID         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    kind                 VARCHAR(40)  NOT NULL
                            CHECK (kind IN (
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
                                'payout_report_resolved'
                            )),
    body                 TEXT         NOT NULL DEFAULT '',
    source_admin_id      UUID                  REFERENCES users (id) ON DELETE SET NULL,
    source_decision_id   UUID                  REFERENCES admin_decision_log (id) ON DELETE SET NULL,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    read_at              TIMESTAMPTZ
);

CREATE INDEX idx_photographer_messages_photographer
    ON photographer_messages (photographer_id, created_at DESC);

CREATE INDEX idx_photographer_messages_unread
    ON photographer_messages (photographer_id, created_at DESC)
    WHERE read_at IS NULL;


CREATE TABLE flags (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_kind         VARCHAR(20)     NOT NULL
                            CHECK (target_kind IN ('photo','user','event')),
    target_id           UUID            NOT NULL,
    reporter_id         UUID                       REFERENCES users (id) ON DELETE SET NULL,
    reason              VARCHAR(40)     NOT NULL,
    note                TEXT            NOT NULL DEFAULT '',
    status              VARCHAR(20)     NOT NULL DEFAULT 'open'
                            CHECK (status IN ('open','resolved','hidden','dismissed')),
    resolution_note     TEXT,
    resolved_by         UUID                       REFERENCES users (id) ON DELETE SET NULL,
    resolved_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX idx_flags_status ON flags (status, created_at DESC);
CREATE INDEX idx_flags_target ON flags (target_kind, target_id);
