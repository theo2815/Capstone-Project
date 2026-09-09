-- V53: payout lifecycle hardening (Phase 2 — race-proof requests, audit-safe cancel)
--
-- 1. Widen the status CHECK to the full auditable lifecycle vocabulary. The
--    PayoutCycleStatus enum is kept a superset of this list. Only `cancelled` is
--    produced yet (soft-cancel of a withdrawn request); approved/processing/
--    failed/returned/rejected are wired by the admin-lifecycle phase.
-- 2. Drop the weekly-model "one cycle per ISO week" unique index. It is wrong for
--    request-based payouts: a photographer paid mid-week must be able to request
--    again that week, and paid/cancelled cycles accumulate. Keeping it would also
--    make request()'s DataIntegrityViolationException -> 409 mapping ambiguous.
-- 3. Replace it with the invariant the request flow actually needs: AT MOST ONE
--    OPEN cycle per photographer. This is the DB guarantee that makes concurrent
--    payout requests unable to double-open (two requests -> one cycle, the loser
--    gets 409). Mirrors uq_payout_reports_open_per_cycle.

ALTER TABLE payout_cycles DROP CONSTRAINT IF EXISTS payout_cycles_status_check;
ALTER TABLE payout_cycles ADD CONSTRAINT payout_cycles_status_check
    CHECK (status IN (
        'scheduled', 'pending', 'approved', 'processing',
        'paid', 'failed', 'returned', 'held', 'rejected', 'cancelled'
    ));

DROP INDEX IF EXISTS uq_payout_cycles_photographer_week;

-- Defensive dedupe so the partial unique index can build on existing data: if a
-- photographer somehow has more than one OPEN cycle (only possible via the
-- pre-fix request race), keep the most recent and cancel the rest. No-op on
-- clean data.
WITH ranked AS (
    SELECT id,
           row_number() OVER (
               PARTITION BY photographer_id
               ORDER BY created_at DESC, id DESC
           ) AS rn
    FROM payout_cycles
    WHERE status IN ('scheduled', 'pending', 'approved', 'processing', 'held')
)
UPDATE payout_cycles c
SET status = 'cancelled'
FROM ranked r
WHERE c.id = r.id AND r.rn > 1;

CREATE UNIQUE INDEX uq_payout_open_per_photographer
    ON payout_cycles (photographer_id)
    WHERE status IN ('scheduled', 'pending', 'approved', 'processing', 'held');
