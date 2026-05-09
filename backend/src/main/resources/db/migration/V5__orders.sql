-- V5: orders / payments / refund pipeline.
--   orders          : one row per (event, checkout) so MockOrder.eventId stays singular.
--                     POST /orders splits the cart by event_id server-side and creates N rows
--                     sharing the same idempotency_key + payment_method + paid_at.
--   order_items     : link orders to photos with the price snapshotted at purchase time
--                     (server-canonical price, never trusted from FE — Q-008 / Q-005).
--   payments        : provider-stub records, future-proof for real GCash / Maya / card webhooks.
--   download_grants : per-photo per-order entitlement window (presigned URL TTL is shorter).
--   disputes        : runner-submitted refund requests, one per photo. Admin queue reads from
--                     this table (Phase G). The mock seed lives in the FE; live mode populates
--                     the real rows here.

CREATE TABLE orders (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID                       REFERENCES users  (id) ON DELETE SET NULL,
    event_id          UUID            NOT NULL   REFERENCES events (id) ON DELETE CASCADE,
    recipient_email   VARCHAR(254)    NOT NULL,
    payment_method    VARCHAR(20)     NOT NULL
                          CHECK (payment_method IN ('gcash', 'maya', 'card')),
    status            VARCHAR(20)     NOT NULL
                          CHECK (status IN ('PENDING', 'PAID', 'FULFILLED', 'REFUNDED')),
    total_php         NUMERIC(12, 2)  NOT NULL,
    idempotency_key   VARCHAR(64),
    paid_at           TIMESTAMPTZ,
    created_at        TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- Idempotent POST /orders per (key, event). Same idempotency-key replayed for the same
-- multi-event cart returns the same N rows; nothing is double-charged.
CREATE UNIQUE INDEX uq_orders_idempotency_key_event
    ON orders (idempotency_key, event_id)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX idx_orders_user_paid_at  ON orders (user_id, paid_at DESC) WHERE user_id IS NOT NULL;
CREATE INDEX idx_orders_recipient     ON orders (recipient_email, paid_at DESC);
CREATE INDEX idx_orders_event_paid_at ON orders (event_id, paid_at DESC);

CREATE TABLE order_items (
    order_id              UUID            NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
    photo_id              UUID            NOT NULL REFERENCES photos (id) ON DELETE RESTRICT,
    price_php_at_purchase NUMERIC(12, 2)  NOT NULL,
    PRIMARY KEY (order_id, photo_id)
);

CREATE INDEX idx_order_items_photo ON order_items (photo_id);

CREATE TABLE payments (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id        UUID            NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
    provider        VARCHAR(20)     NOT NULL
                        CHECK (provider IN ('gcash', 'maya', 'card', 'stub')),
    provider_ref    VARCHAR(100),
    amount_php      NUMERIC(12, 2)  NOT NULL,
    status          VARCHAR(20)     NOT NULL
                        CHECK (status IN ('PENDING', 'SUCCEEDED', 'FAILED')),
    paid_at         TIMESTAMPTZ,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX idx_payments_order        ON payments (order_id);
CREATE UNIQUE INDEX uq_payments_provider_ref
    ON payments (provider, provider_ref)
    WHERE provider_ref IS NOT NULL;

CREATE TABLE download_grants (
    order_id            UUID         NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
    photo_id            UUID         NOT NULL REFERENCES photos (id) ON DELETE CASCADE,
    granted_until       TIMESTAMPTZ  NOT NULL,
    last_downloaded_at  TIMESTAMPTZ,
    download_count      INTEGER      NOT NULL DEFAULT 0,
    PRIMARY KEY (order_id, photo_id)
);

CREATE INDEX idx_download_grants_photo ON download_grants (photo_id);

CREATE TABLE disputes (
    id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id          UUID            NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
    photo_id          UUID            NOT NULL REFERENCES photos (id) ON DELETE CASCADE,
    runner_id         UUID                       REFERENCES users  (id) ON DELETE SET NULL,
    reason            VARCHAR(40)     NOT NULL
                          CHECK (reason IN ('wrong_runner', 'low_quality', 'not_received',
                                            'duplicate_charge', 'other')),
    note              TEXT            NOT NULL DEFAULT '',
    status            VARCHAR(20)     NOT NULL DEFAULT 'open'
                          CHECK (status IN ('open', 'resolved', 'denied', 'escalated')),
    resolution        VARCHAR(20)
                          CHECK (resolution IN ('refund_full', 'refund_partial', 'deny')),
    refund_amount_php NUMERIC(12, 2),
    opened_at         TIMESTAMPTZ     NOT NULL DEFAULT now(),
    resolved_at       TIMESTAMPTZ
);

CREATE INDEX idx_disputes_order  ON disputes (order_id);
CREATE INDEX idx_disputes_photo  ON disputes (photo_id);
CREATE INDEX idx_disputes_status ON disputes (status, opened_at DESC);
CREATE INDEX idx_disputes_runner ON disputes (runner_id, opened_at DESC) WHERE runner_id IS NOT NULL;

-- One open / escalated dispute per (order, photo). Once resolved or denied the runner
-- can still re-submit (per refund-helpers.ts: denied disputes are re-submittable).
CREATE UNIQUE INDEX uq_disputes_open_per_order_photo
    ON disputes (order_id, photo_id)
    WHERE status IN ('open', 'escalated');
