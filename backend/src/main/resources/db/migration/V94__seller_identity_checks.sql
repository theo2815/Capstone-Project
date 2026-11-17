-- AA-I1: automatic seller identity checks. SUBMITTED becomes MANUAL_REVIEW, AUTO_APPROVED is new,
-- and the columns later tasks fill are added. Safe on an empty table and on rows written under V93.

-- 1-3. Status values. No IF EXISTS: a different constraint name must fail loudly.
ALTER TABLE photographer_seller_details DROP CONSTRAINT photographer_seller_details_status_check;
UPDATE photographer_seller_details SET status = 'MANUAL_REVIEW' WHERE status = 'SUBMITTED';
ALTER TABLE photographer_seller_details
    ADD CONSTRAINT chk_seller_details_status
        CHECK (status IN ('MANUAL_REVIEW', 'AUTO_APPROVED', 'APPROVED', 'REJECTED'));

-- 4-5. What the automatic checks found (scores on AWS's 0-100 scale).
ALTER TABLE photographer_seller_details
    ADD COLUMN liveness_session_id    VARCHAR(128),
    ADD COLUMN liveness_confidence    NUMERIC(5,2),
    ADD COLUMN face_similarity        NUMERIC(5,2),
    ADD COLUMN auto_checked_at        TIMESTAMPTZ,
    ADD COLUMN biometric_consent_at   TIMESTAMPTZ,
    ADD COLUMN liveness_reference_key VARCHAR(512),
    ADD COLUMN id_number_hmac         VARCHAR(64),
    ADD COLUMN reason_codes           JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN repeat_matches         JSONB NOT NULL DEFAULT '[]';

-- 6. The phone's last 10 digits, for repeat-account matching (the phone is stored as typed).
ALTER TABLE photographer_seller_details ADD COLUMN phone_key VARCHAR(10);
UPDATE photographer_seller_details SET phone_key = right(regexp_replace(phone, '\D', '', 'g'), 10);
ALTER TABLE photographer_seller_details ALTER COLUMN phone_key SET NOT NULL;

-- 7. Lookups for repeat-account matching.
CREATE INDEX idx_seller_details_id_number_hmac ON photographer_seller_details (id_number_hmac);
CREATE INDEX idx_seller_details_phone_key ON photographer_seller_details (phone_key);
CREATE INDEX idx_payout_accounts_account_number ON photographer_payout_accounts (account_number);

-- 8. Single-use Face Liveness sessions. Shared with the runner photo-removal plan (RD-0.1):
-- a session is bound to one purpose, and to one event for runner removals only.
CREATE TABLE liveness_sessions (
    session_id      VARCHAR(128) PRIMARY KEY,
    user_id         UUID         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    purpose         VARCHAR(30)  NOT NULL
                      CHECK (purpose IN ('PHOTOGRAPHER_IDENTITY', 'RUNNER_PHOTO_REMOVAL')),
    event_id        UUID         REFERENCES events (id),
    consent_version VARCHAR(16),
    created_at      TIMESTAMPTZ  NOT NULL,
    consumed_at     TIMESTAMPTZ,
    CONSTRAINT chk_liveness_sessions_event_for_removal
        CHECK ((purpose = 'RUNNER_PHOTO_REMOVAL') = (event_id IS NOT NULL))
);
CREATE INDEX idx_liveness_sessions_user_created ON liveness_sessions (user_id, created_at);
