-- V55: payout lifecycle audit trail (Phase 4a).
--
-- Append-only history of every payout-cycle status transition, by any actor
-- (photographer / admin / system / provider). Answers "what happened to this
-- payout, when, and who did it" for reconciliation and disputes — complements
-- admin_decision_log (admin actions only) by also capturing photographer
-- (request/cancel) and, later, provider (webhook) transitions.

CREATE TABLE payout_status_history (
    id               UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    payout_cycle_id  VARCHAR(80)   NOT NULL REFERENCES payout_cycles (id) ON DELETE CASCADE,
    from_status      VARCHAR(20),
    to_status        VARCHAR(20)   NOT NULL,
    actor_type       VARCHAR(20)   NOT NULL
                         CHECK (actor_type IN ('photographer', 'admin', 'system', 'provider')),
    actor_id         UUID,
    reason           VARCHAR(255),
    created_at       TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX idx_payout_status_history_cycle
    ON payout_status_history (payout_cycle_id, created_at);
