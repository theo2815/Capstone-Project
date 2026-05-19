-- V20 — Runner-side inbox. Parallel to photographer_messages (V10 + V15)
-- so admin actions that target a runner (dispute outcome, suspension,
-- admin DM) can write a row in the same TX as the decision-log entry.
--
-- Kept structurally identical to photographer_messages except:
--   - runner_id (FK users) instead of photographer_id
--   - order_id (FK orders, NULLABLE) — most runner notifications are
--     dispute outcomes which deep-link to /orders?expand={orderId}. Set
--     null for account_suspended / admin_message kinds.
--   - CHECK constraint is the runner-relevant subset of kinds (no
--     verification_*, payout_*, force_edit — those are photographer-only).
--
-- ON DELETE: runner_id CASCADE so a wipe-user run clears their inbox.
--            source_admin_id / source_decision_id SET NULL — keep the
--            row but lose the pointer when admin or audit row vanishes.
--            order_id SET NULL — keep the notification even if the order
--            is later deleted; orderId becomes null and the FE falls
--            back to "View order" → orders list.

CREATE TABLE runner_messages (
    id                   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    runner_id            UUID         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    kind                 VARCHAR(40)  NOT NULL
                            CHECK (kind IN (
                                'dispute_resolved',
                                'dispute_denied',
                                'dispute_escalated',
                                'admin_message',
                                'account_suspended',
                                'account_unsuspended'
                            )),
    title                TEXT,
    body                 TEXT         NOT NULL DEFAULT '',
    source_admin_id      UUID                  REFERENCES users (id) ON DELETE SET NULL,
    source_decision_id   UUID                  REFERENCES admin_decision_log (id) ON DELETE SET NULL,
    order_id             UUID                  REFERENCES orders (id) ON DELETE SET NULL,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    read_at              TIMESTAMPTZ,
    removed_at           TIMESTAMPTZ
);

CREATE INDEX idx_runner_messages_runner
    ON runner_messages (runner_id, created_at DESC)
    WHERE removed_at IS NULL;

CREATE INDEX idx_runner_messages_unread
    ON runner_messages (runner_id, created_at DESC)
    WHERE read_at IS NULL AND removed_at IS NULL;
