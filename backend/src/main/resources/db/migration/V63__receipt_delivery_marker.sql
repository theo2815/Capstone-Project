ALTER TABLE orders
    ADD COLUMN receipt_delivered_at TIMESTAMPTZ;

UPDATE orders
SET receipt_delivered_at = email_sent_at
WHERE email_sent_at IS NOT NULL;

CREATE INDEX idx_orders_stale_receipt_claims
    ON orders (email_sent_at)
    WHERE email_sent_at IS NOT NULL AND receipt_delivered_at IS NULL;
