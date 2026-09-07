-- V56: real admin payout lifecycle columns (Phase 4b).
--
-- Backs the explicit state machine (approve/reject/processing/paid/failed/
-- returned) so "approved" stops being derived from PENDING+settledAt. All
-- nullable and additive. second_approved_by is reserved for Phase 4c
-- (two-person approval).

ALTER TABLE payout_cycles
    ADD COLUMN approved_by        UUID,
    ADD COLUMN approved_at        TIMESTAMPTZ,
    ADD COLUMN second_approved_by UUID,
    ADD COLUMN processing_at      TIMESTAMPTZ,
    ADD COLUMN paid_by            UUID,
    ADD COLUMN failed_reason      VARCHAR(255),
    ADD COLUMN returned_at        TIMESTAMPTZ,
    ADD COLUMN rejected_reason    VARCHAR(255);
