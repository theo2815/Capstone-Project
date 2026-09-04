ALTER TABLE orders DROP CONSTRAINT orders_payment_method_check;
ALTER TABLE orders ADD CONSTRAINT orders_payment_method_check
    CHECK (payment_method IN ('gcash', 'maya', 'card', 'qrph'));

ALTER TABLE payments ADD COLUMN expires_at TIMESTAMPTZ;
