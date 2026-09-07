-- V58: disbursement seam + immutable destination snapshot (Phase 8)
--
-- 1. provider / provider_transfer_id: which rail moved the money and its id at
--    that provider. 'manual' = admin sends by hand (today); 'paymongo' = a
--    Transfer created via the API (Phase 11, disabled by default). Stamped when
--    disbursement is initiated (mark-processing).
-- 2. account_snapshot_*: the destination captured onto the cycle at REQUEST time
--    so a later payout-account change can't silently reroute an in-flight payout,
--    and the disbursement targets a fixed account. Immutable once set.
--
-- All additive + nullable/defaulted; no destructive change.

ALTER TABLE payout_cycles
    ADD COLUMN provider VARCHAR(20) NOT NULL DEFAULT 'manual',
    ADD COLUMN provider_transfer_id VARCHAR(100),
    ADD COLUMN account_snapshot_method VARCHAR(20),
    ADD COLUMN account_snapshot_number VARCHAR(64),
    ADD COLUMN account_snapshot_name VARCHAR(200);

-- Best-effort backfill for still-open cycles: snapshot from the photographer's
-- current primary payout account so an in-flight cycle has a fixed destination.
-- Terminal cycles (paid/failed/returned/rejected/cancelled) are historical and
-- left null. No-op where a photographer has no primary account.
UPDATE payout_cycles c
SET account_snapshot_method = lower(a.method),
    account_snapshot_number = a.account_number,
    account_snapshot_name   = a.account_name
FROM photographer_payout_accounts a
WHERE a.user_id = c.photographer_id
  AND a.is_primary = TRUE
  AND c.account_snapshot_number IS NULL
  AND c.status IN ('scheduled', 'pending', 'approved', 'processing', 'held');
