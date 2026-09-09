-- Preserve approvals granted before APPROVED became a stored state. Existing
-- approvals are grandfathered; new approvals follow the two-person policy.
UPDATE payout_cycles
SET status = 'approved', approved_at = COALESCE(approved_at, settled_at), settled_at = NULL
WHERE status = 'pending' AND settled_at IS NOT NULL;

-- Do not backfill QR from today's account: its historical recipient is unknown.
-- Existing cycles retain their frozen number/name; new requests freeze the QR too.
ALTER TABLE payout_cycles ADD COLUMN account_snapshot_qr_key VARCHAR(512);
CREATE INDEX ix_payout_snapshot_qr ON payout_cycles(account_snapshot_qr_key)
    WHERE account_snapshot_qr_key IS NOT NULL;

ALTER TABLE admin_mfa ADD COLUMN last_used_step BIGINT;
