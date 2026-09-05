-- V50: attribute coupons per order item. Checkout now applies every live
-- coupon of the (event, photographer) pairs in the cart, so one event order
-- can carry coupons from several photographers. orders.coupon_id / coupon_code
-- stay as the snapshot of the code the runner typed (idempotent replay,
-- receipts); usage limits count distinct orders through order_items.

ALTER TABLE order_items
    ADD COLUMN coupon_id UUID,
    ADD CONSTRAINT fk_order_items_coupon
        FOREIGN KEY (coupon_id) REFERENCES photographer_coupons (id) ON DELETE SET NULL;

CREATE INDEX idx_order_items_coupon
    ON order_items (coupon_id)
    WHERE coupon_id IS NOT NULL;

-- Pre-V50 orders held one coupon per event order; the discounted items are
-- the ones it reached.
UPDATE order_items oi
SET coupon_id = o.coupon_id
FROM orders o
WHERE oi.order_id = o.id
  AND o.coupon_id IS NOT NULL
  AND oi.discount_php > 0;
