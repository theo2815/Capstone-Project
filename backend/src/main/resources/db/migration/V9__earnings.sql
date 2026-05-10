-- V9: earnings + billing pipeline (PR 9 / Phase F.3)
--   transactions   : ledger of photographer earnings keyed off paid orders.
--                    Per Q-E3 a refund is materialized as a separate row with a
--                    negative amount_kept_php and refund_of pointing to the
--                    originating tx — keeps GROUP BY math correct without a
--                    parallel "refunds" table or signed-status enum.
--   payout_cycles  : weekly photographer payout buckets. id is the human-readable
--                    `PAY-{YYYY}W{NN}-{HANDLE}` string per Q-A1 so the FE can
--                    surface it directly in receipts and report URLs without an
--                    extra round-trip. Cycle generator is icebox for PR 9 — admin
--                    or a future cron will populate rows; the read endpoints
--                    return whatever is there (empty list when there is nothing).
--   payout_reports : photographer-filed reports against a cycle. Partial unique
--                    index enforces at-most-one OPEN report per (photographer,
--                    cycle) so the FE's REPORT_ALREADY_OPEN gate is mirrored
--                    in the DB rather than a service-only race.

CREATE TABLE transactions (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Snapshot of the paid_at instant from the originating order so weekly /
    -- monthly rollups don't drift if rows are inserted retroactively.
    paid_at             TIMESTAMPTZ     NOT NULL,
    photographer_id     UUID            NOT NULL REFERENCES users    (id) ON DELETE CASCADE,
    event_id            UUID            NOT NULL REFERENCES events   (id) ON DELETE CASCADE,
    photo_id            UUID            NOT NULL REFERENCES photos   (id) ON DELETE CASCADE,
    order_id            UUID            NOT NULL REFERENCES orders   (id) ON DELETE CASCADE,
    buyer_id            UUID                       REFERENCES users   (id) ON DELETE SET NULL,
    -- Snapshot of the buyer's privacy-safe display ("Aira S.") at txn time so
    -- the ledger keeps reading correctly even if the buyer renames or deletes.
    buyer_display_name  VARCHAR(100)    NOT NULL DEFAULT '',
    amount_kept_php     NUMERIC(12, 2)  NOT NULL,
    is_refund           BOOLEAN         NOT NULL DEFAULT false,
    refund_of           UUID                       REFERENCES transactions (id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- One sale (or one refund) per (order, photo). Lets us no-op on retried order
-- creates without producing duplicate ledger rows.
CREATE UNIQUE INDEX uq_transactions_order_photo_refund
    ON transactions (order_id, photo_id, is_refund);

CREATE INDEX idx_transactions_photographer_paid_at
    ON transactions (photographer_id, paid_at DESC);

CREATE INDEX idx_transactions_photographer_event
    ON transactions (photographer_id, event_id);

CREATE INDEX idx_transactions_buyer
    ON transactions (buyer_id, paid_at DESC) WHERE buyer_id IS NOT NULL;


CREATE TABLE payout_cycles (
    id                 VARCHAR(80)     PRIMARY KEY,
    photographer_id    UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    week_of            DATE            NOT NULL,
    amount_php         NUMERIC(14, 2)  NOT NULL DEFAULT 0,
    item_count         INTEGER         NOT NULL DEFAULT 0,
    status             VARCHAR(20)     NOT NULL DEFAULT 'scheduled'
                          CHECK (status IN ('scheduled','pending','paid','held')),
    method             VARCHAR(20)     NOT NULL
                          CHECK (method IN ('gcash','maya','gotyme')),
    settled_at         TIMESTAMPTZ,
    payment_reference  VARCHAR(100),
    hold_reason        VARCHAR(255),
    created_at         TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- One cycle per photographer per ISO week.
CREATE UNIQUE INDEX uq_payout_cycles_photographer_week
    ON payout_cycles (photographer_id, week_of);

CREATE INDEX idx_payout_cycles_photographer_week_desc
    ON payout_cycles (photographer_id, week_of DESC);


CREATE TABLE payout_reports (
    id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    payout_id          VARCHAR(80)     NOT NULL REFERENCES payout_cycles (id) ON DELETE CASCADE,
    photographer_id    UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    reason             VARCHAR(40)     NOT NULL
                          CHECK (reason IN ('not_received','wrong_amount','account_info','processing_delay','other')),
    note               TEXT            NOT NULL DEFAULT '',
    status             VARCHAR(20)     NOT NULL DEFAULT 'open'
                          CHECK (status IN ('open','acknowledged','resolved')),
    opened_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    acknowledged_at    TIMESTAMPTZ,
    acknowledge_reply  TEXT,
    resolved_at        TIMESTAMPTZ,
    resolution_note    TEXT
);

-- At most one OPEN report per (cycle, photographer). Subsequent reports against
-- the same cycle are only allowed once the prior open report flips to
-- acknowledged or resolved — keeps the FE button-state logic and the DB in sync.
CREATE UNIQUE INDEX uq_payout_reports_open_per_cycle
    ON payout_reports (payout_id, photographer_id)
    WHERE status = 'open';

CREATE INDEX idx_payout_reports_photographer_opened
    ON payout_reports (photographer_id, opened_at DESC);

CREATE INDEX idx_payout_reports_status
    ON payout_reports (status, opened_at DESC);
