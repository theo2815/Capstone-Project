-- V11: idempotency for POST /api/v1/admin/payouts/bulk (C-3).
--
-- Plan: .claude-qp/plans/alright-claude-i-recently-ancient-milner.md (§2 C-3).
--
-- Problem: AdminPayoutService.bulk() generates a fresh UUID groupId on every
--   call, so a network-blip retry of the same admin intent (Idempotency-Key
--   header) produces a second groupId, a second admin_decision_log row per
--   cycle, AND a second photographer_messages row per cycle. The
--   photographer's inbox spams duplicate "your payout was approved"
--   notifications; the audit trail looks like the admin clicked the button
--   twice; KPI counts inflate.
--
-- Fix: stamp the Idempotency-Key on every admin_decision_log row in the bulk
--   run. Two indexes:
--
--   uq_admin_decision_log_admin_idem_target_payout
--     Per-row uniqueness on (admin_id, idempotency_key, target_payout_id)
--     so a duplicate bulk-payout decision for the same (admin, key, cycle)
--     can never land twice. Partial — keyed only when all three are
--     non-null, so the column stays optional for non-bulk decision rows
--     (logUserDecision / logEventDecision / logDisputeDecision currently
--     don't take an idempotency key).
--
--   idx_admin_decision_log_admin_idem
--     Non-unique lookup index for the AdminPayoutService.bulk fast-path:
--     `findGroupIdByAdminAndIdempotencyKey(adminId, key) → existing
--     groupId`, which lets a retry reuse the original groupId (and skip
--     re-issuing inbox messages on each per-row gate).
--
-- Backfill: not required. Pre-V11 admin_decision_log rows have no
--   idempotency_key (NULL is the new "we don't know" default), and the
--   partial unique only guards rows that actually opt in.

ALTER TABLE admin_decision_log
    ADD COLUMN idempotency_key VARCHAR(64);

CREATE UNIQUE INDEX uq_admin_decision_log_admin_idem_target_payout
    ON admin_decision_log (admin_id, idempotency_key, target_payout_id)
    WHERE idempotency_key IS NOT NULL AND target_payout_id IS NOT NULL;

CREATE INDEX idx_admin_decision_log_admin_idem
    ON admin_decision_log (admin_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;
