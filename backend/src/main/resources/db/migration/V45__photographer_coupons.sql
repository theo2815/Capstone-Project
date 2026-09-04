-- V45: photographer coupons (2026-09-04)
--   photographer_coupons : one code per photographer (PK = photographer_id).
--                          A coupon is a percentage of the photographer's share
--                          (list price × keep rate), never of the list price, so
--                          QuickPitik's cut on a sale is unchanged by any code.
--                          Codes are stored UPPERCASE and are globally unique
--                          because a runner types only the code at checkout.
--                          The 1–50 cap lives in config
--                          (app.platform.coupon-max-percent); the CHECK below is
--                          a sanity bound, not the business rule.
--   orders.coupon_code        : the code entered at checkout, stamped on every
--                               order row of that checkout so an idempotent
--                               replay can compare it without re-resolving.
--   order_items.discount_php  : per-item discount (0 when the item was not
--                               eligible). price_php_at_purchase stays the list
--                               price; what the runner paid is price − discount.
--   transactions.discount_php : ledger copy of the discount. Admin KPIs used to
--                               reconstruct gross as kept / keep_rate, which
--                               under-reports the platform fee once a discount
--                               exists; with this column gross is
--                               (kept + discount) / keep_rate. Refund rows carry
--                               the negative, exactly like amount_kept_php.

CREATE TABLE photographer_coupons (
    photographer_id UUID         PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    code            VARCHAR(16)  NOT NULL,
    percent_off     INTEGER      NOT NULL CHECK (percent_off BETWEEN 1 AND 100),
    active          BOOLEAN      NOT NULL DEFAULT true,
    expires_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX uq_photographer_coupons_code ON photographer_coupons (code);

ALTER TABLE orders       ADD COLUMN coupon_code  VARCHAR(16);
ALTER TABLE order_items  ADD COLUMN discount_php NUMERIC(12, 2) NOT NULL DEFAULT 0;
ALTER TABLE transactions ADD COLUMN discount_php NUMERIC(12, 2) NOT NULL DEFAULT 0;
