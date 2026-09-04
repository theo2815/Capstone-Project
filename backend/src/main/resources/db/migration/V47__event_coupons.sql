-- V47: scope photographer coupons to one owned event and make optional usage
-- limits concurrency-safe through persisted orders.

ALTER TABLE events
    ADD CONSTRAINT uq_events_id_created_by UNIQUE (id, created_by);

ALTER TABLE photographer_coupons
    ADD COLUMN id          UUID DEFAULT gen_random_uuid(),
    ADD COLUMN event_id    UUID,
    ADD COLUMN usage_limit INTEGER CHECK (usage_limit IS NULL OR usage_limit > 0);

ALTER TABLE photographer_coupons DROP CONSTRAINT photographer_coupons_pkey;

ALTER TABLE photographer_coupons
    ALTER COLUMN id SET NOT NULL,
    ADD CONSTRAINT photographer_coupons_pkey PRIMARY KEY (id),
    ADD CONSTRAINT fk_photographer_coupons_owned_event
        FOREIGN KEY (event_id, photographer_id)
        REFERENCES events (id, created_by)
        ON DELETE CASCADE;

-- V45 rows had no event to authorize. Preserve them for audit, but never let
-- an unscoped row surface or redeem after this migration.
UPDATE photographer_coupons SET active = false WHERE event_id IS NULL;

ALTER TABLE photographer_coupons
    ADD CONSTRAINT photographer_coupons_active_event_check
        CHECK (event_id IS NOT NULL OR active = false);

DROP INDEX uq_photographer_coupons_code;
CREATE UNIQUE INDEX uq_photographer_coupons_code
    ON photographer_coupons (code)
    WHERE event_id IS NOT NULL;
CREATE UNIQUE INDEX uq_photographer_coupons_event
    ON photographer_coupons (event_id)
    WHERE event_id IS NOT NULL;
CREATE INDEX idx_photographer_coupons_photographer
    ON photographer_coupons (photographer_id);

ALTER TABLE orders
    ADD COLUMN coupon_id UUID REFERENCES photographer_coupons (id) ON DELETE SET NULL;

CREATE INDEX idx_orders_coupon_usage
    ON orders (coupon_id, status)
    WHERE coupon_id IS NOT NULL;
