-- V19: extend `disputes.status` with 'withdrawn' so a runner can cancel an
-- open refund request before admin touches it. The partial unique index
-- `uq_disputes_open_per_order_photo` covers ('open','escalated') only, so
-- a withdrawn dispute leaves the (order, photo) slot free for re-submission
-- — matches the runner-side helper in refund-helpers.ts (denied disputes
-- are re-submittable, and now withdrawn ones are too).
--
-- Cancel policy lives in RefundService.withdraw: OPEN-only. Once admin
-- has ESCALATED the dispute, the runner can no longer cancel — admin has
-- already invested review time and the resolution outcome belongs to them.

ALTER TABLE disputes
    DROP CONSTRAINT disputes_status_check;

ALTER TABLE disputes
    ADD CONSTRAINT disputes_status_check
        CHECK (status IN ('open', 'resolved', 'denied', 'escalated', 'withdrawn'));

ALTER TABLE disputes
    ADD COLUMN withdrawn_at TIMESTAMPTZ;
