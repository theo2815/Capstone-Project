-- Runner photo deletion requests (RD-1): a RUNNER source on the existing removal case model, plus the
-- email claim columns RD-9/RD-10 use and a per-photo face count. Every new column is nullable or defaulted,
-- so code that does not know them keeps working. Old CHECKs are dropped by their Postgres default names,
-- with no IF EXISTS, so a wrong name fails this migration loudly.

ALTER TABLE photo_removal_requests
    ADD COLUMN source VARCHAR(16) NOT NULL DEFAULT 'SUPPORT' CHECK (source IN ('SUPPORT', 'RUNNER')),
    ADD COLUMN reason_code VARCHAR(32),
    ADD COLUMN consent_version VARCHAR(16),
    ADD COLUMN consent_at TIMESTAMPTZ,
    ADD COLUMN liveness_session_id VARCHAR(128),
    ADD COLUMN liveness_band VARCHAR(16) CHECK (liveness_band IN ('PASS', 'UNCERTAIN')),
    ADD COLUMN liveness_confidence NUMERIC(5, 2),
    ADD COLUMN risk_flags JSONB NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(risk_flags) = 'array'),
    ADD COLUMN close_reason VARCHAR(16) CHECK (close_reason IN ('NO_MATCHES', 'WITHDRAWN'));

-- One live runner request per runner and event. A withdrawn or no-match case waiting in RESOLVING for its
-- evidence purge never blocks a new one; a decided case finishing its erasure still does.
CREATE UNIQUE INDEX uq_photo_removal_requests_runner_open ON photo_removal_requests (requester_user_id, event_id_snapshot)
    WHERE source = 'RUNNER' AND (status IN ('DRAFT', 'OPEN') OR (status = 'RESOLVING' AND close_reason IS NULL));
CREATE UNIQUE INDEX uq_photo_removal_requests_liveness_session ON photo_removal_requests (liveness_session_id)
    WHERE liveness_session_id IS NOT NULL;
CREATE INDEX idx_photo_removal_requests_requester ON photo_removal_requests (requester_user_id, created_at DESC)
    WHERE source = 'RUNNER';

ALTER TABLE photo_removal_history DROP CONSTRAINT photo_removal_history_actor_type_check;
ALTER TABLE photo_removal_history ADD CONSTRAINT photo_removal_history_actor_type_check
    CHECK (actor_type IN ('ADMIN', 'PHOTOGRAPHER', 'SYSTEM', 'SUPPORT', 'RUNNER'));
ALTER TABLE photo_removal_history DROP CONSTRAINT photo_removal_history_action_check;
ALTER TABLE photo_removal_history ADD CONSTRAINT photo_removal_history_action_check
    CHECK (action IN ('CREATED', 'EVIDENCE_REPLACED', 'VERIFIED', 'NOTICE_STORED', 'OWNER_UNAVAILABLE',
        'OWNER_RESPONDED', 'REVIEW_OVERDUE', 'ADMIN_DECIDED', 'ERASURE_PROGRESS', 'EVIDENCE_PURGED', 'SUPPORT_UPDATED',
        'CANCEL_REQUESTED', 'RESOLVED', 'CANCELLED', 'SUGGESTED', 'SUGGESTION_DISMISSED'));

ALTER TABLE photo_removal_commands DROP CONSTRAINT photo_removal_commands_operation_check;
ALTER TABLE photo_removal_commands ADD CONSTRAINT photo_removal_commands_operation_check
    CHECK (operation IN ('CREATE', 'REPLACE_EVIDENCE', 'CANCEL', 'MATCH_CANDIDATES', 'VERIFY', 'RESPOND',
        'DECIDE', 'RETRY_CLEANUP', 'SUPPORT_UPDATE', 'REVIEW_SUGGESTIONS'));

ALTER TABLE runner_messages DROP CONSTRAINT runner_messages_kind_check;
ALTER TABLE runner_messages ADD CONSTRAINT runner_messages_kind_check CHECK (kind IN (
    'dispute_resolved', 'dispute_denied', 'dispute_escalated', 'admin_message',
    'account_suspended', 'account_unsuspended', 'photo_removed', 'photo_removal_update'
));

-- Email send claim and delivery marker (the orders.email_sent_at + receipt_delivered_at pattern, V63).
ALTER TABLE runner_messages ADD COLUMN email_sent_at TIMESTAMPTZ, ADD COLUMN email_delivered_at TIMESTAMPTZ;
ALTER TABLE photographer_messages ADD COLUMN email_sent_at TIMESTAMPTZ, ADD COLUMN email_delivered_at TIMESTAMPTZ;

-- Faces the provider indexed; a lower bound (small or blurred faces are not kept). NULL = not counted.
ALTER TABLE photos ADD COLUMN face_count SMALLINT CHECK (face_count IS NULL OR face_count >= 0);
