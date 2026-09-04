-- Scope checkout idempotency to the buyer, add a terminal abandoned-checkout
-- state, replace stored bearer tokens with one-way legacy hashes, and retain
-- the provider identifiers needed to issue and reconcile real refunds.

DROP INDEX IF EXISTS uq_orders_idempotency_key_event;

CREATE UNIQUE INDEX uq_orders_user_idempotency_key_event
    ON orders (user_id, idempotency_key, event_id)
    WHERE user_id IS NOT NULL AND idempotency_key IS NOT NULL;

CREATE UNIQUE INDEX uq_orders_guest_idempotency_key_event
    ON orders (lower(recipient_email), idempotency_key, event_id)
    WHERE user_id IS NULL AND idempotency_key IS NOT NULL;

ALTER TABLE orders DROP CONSTRAINT orders_status_check;
ALTER TABLE orders ADD CONSTRAINT orders_status_check
    CHECK (status IN ('PENDING', 'PAID', 'FULFILLED', 'REFUNDED', 'EXPIRED'));

UPDATE orders o
SET status = 'FULFILLED'
WHERE o.status = 'PAID'
  AND EXISTS (SELECT 1 FROM download_grants dg WHERE dg.order_id = o.id);

ALTER TABLE orders DROP CONSTRAINT uq_orders_share_token;
UPDATE orders
SET share_token = encode(digest(share_token, 'sha256'), 'hex')
WHERE share_token IS NOT NULL;
ALTER TABLE orders RENAME COLUMN share_token TO legacy_share_token_hash;
ALTER TABLE orders ADD CONSTRAINT uq_orders_legacy_share_token_hash
    UNIQUE (legacy_share_token_hash);

ALTER TABLE payments
    ADD COLUMN provider_payment_id VARCHAR(100);

CREATE UNIQUE INDEX uq_payments_provider_payment_order
    ON payments (provider, provider_payment_id, order_id)
    WHERE provider_payment_id IS NOT NULL;

ALTER TABLE disputes
    ADD COLUMN provider_refund_id  VARCHAR(100),
    ADD COLUMN refund_status       VARCHAR(20),
    ADD COLUMN refund_requested_at TIMESTAMPTZ,
    ADD COLUMN refunded_at         TIMESTAMPTZ,
    ADD COLUMN refund_requested_by UUID REFERENCES users (id) ON DELETE SET NULL,
    ADD COLUMN refund_reason       TEXT;

CREATE UNIQUE INDEX uq_disputes_provider_refund_id
    ON disputes (provider_refund_id)
    WHERE provider_refund_id IS NOT NULL;
