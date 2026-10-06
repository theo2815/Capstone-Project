ALTER TABLE photo_removal_commands DROP CONSTRAINT photo_removal_commands_operation_check;
ALTER TABLE photo_removal_commands ADD CONSTRAINT photo_removal_commands_operation_check
    CHECK (operation IN ('CREATE', 'REPLACE_EVIDENCE', 'CANCEL', 'MATCH_CANDIDATES', 'VERIFY'));

ALTER TABLE photographer_messages DROP CONSTRAINT photographer_messages_kind_check;
ALTER TABLE photographer_messages ADD CONSTRAINT photographer_messages_kind_check CHECK (kind IN (
    'verification_approved', 'verification_rejected', 'verification_reset', 'suspended', 'unsuspended',
    'force_edit', 'dispute_resolved', 'dispute_denied', 'dispute_escalated', 'payout_approved', 'payout_held',
    'payout_paid', 'payout_report_acknowledged', 'payout_report_resolved', 'admin_message', 'event_approved',
    'event_rejected', 'event_change_approved', 'event_change_rejected', 'photo_removal_request', 'photo_removal_outcome'
));
ALTER TABLE photographer_messages
    ADD COLUMN photo_removal_request_id UUID REFERENCES photo_removal_requests(id),
    ADD COLUMN notice_key VARCHAR(160);
CREATE UNIQUE INDEX uq_photographer_messages_notice ON photographer_messages(notice_key) WHERE notice_key IS NOT NULL;

-- Outcome delivery writers are introduced by their owning later tasks.
ALTER TABLE runner_messages DROP CONSTRAINT runner_messages_kind_check;
ALTER TABLE runner_messages ADD CONSTRAINT runner_messages_kind_check CHECK (kind IN (
    'dispute_resolved', 'dispute_denied', 'dispute_escalated', 'admin_message',
    'account_suspended', 'account_unsuspended', 'photo_removed'
));
ALTER TABLE runner_messages
    ADD COLUMN photo_removal_request_id UUID REFERENCES photo_removal_requests(id),
    ADD COLUMN notice_key VARCHAR(160);
CREATE UNIQUE INDEX uq_runner_messages_notice ON runner_messages(notice_key) WHERE notice_key IS NOT NULL;

ALTER TABLE photo_removal_reviews ADD COLUMN notification_next_attempt_at TIMESTAMPTZ;
CREATE INDEX idx_photo_removal_reviews_pending_notice ON photo_removal_reviews(notification_next_attempt_at, id)
    WHERE notified_at IS NULL AND unavailable_reason IS NULL;
