-- V54: payout-account change audit + anti-fraud cooldown source (Phase 3a).
--
-- Append-only trail of every payout-destination change (create / update /
-- set-primary / delete / qr). The most recent row per user also starts the
-- post-change payout cooldown that PayoutRequestService enforces, so no separate
-- last_changed_at column is needed. Account numbers are stored MASKED (last 4) —
-- this is a forensic trail, not a second copy of the sensitive number.

CREATE TABLE payout_account_audit (
    id                  UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID          NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    action              VARCHAR(20)   NOT NULL
                            CHECK (action IN ('create', 'update', 'set_primary', 'delete', 'qr')),
    method              VARCHAR(20),
    account_masked_old  VARCHAR(40),
    account_masked_new  VARCHAR(40),
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- Serves both the cooldown existence check (user_id + created_at >= cutoff) and
-- the "latest change" lookup (user_id, created_at DESC).
CREATE INDEX idx_payout_account_audit_user_created
    ON payout_account_audit (user_id, created_at DESC);
