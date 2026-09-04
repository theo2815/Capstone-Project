-- V49: coupons follow coverage (event_photographer), not event ownership.
-- V47 tied a coupon to events(id, created_by), which made admin-created events
-- (created_by IS NULL) impossible to discount for the photographers who shot
-- them. Coverage is the same predicate checkout already uses (photo.event_id
-- + photo.photographer_id), so one coupon per (event, photographer).

ALTER TABLE photographer_coupons DROP CONSTRAINT fk_photographer_coupons_owned_event;
ALTER TABLE events DROP CONSTRAINT uq_events_id_created_by;

-- Every live coupon must have a coverage row before the FK lands. Owned events
-- save one at creation; this only backfills a row that somehow lacks it.
INSERT INTO event_photographer (event_id, photographer_id)
SELECT event_id, photographer_id FROM photographer_coupons WHERE event_id IS NOT NULL
ON CONFLICT DO NOTHING;

ALTER TABLE photographer_coupons
    ADD CONSTRAINT fk_photographer_coupons_coverage
        FOREIGN KEY (event_id, photographer_id)
        REFERENCES event_photographer (event_id, photographer_id)
        ON DELETE CASCADE;

DROP INDEX uq_photographer_coupons_event;
CREATE UNIQUE INDEX uq_photographer_coupons_event_photographer
    ON photographer_coupons (event_id, photographer_id)
    WHERE event_id IS NOT NULL;
