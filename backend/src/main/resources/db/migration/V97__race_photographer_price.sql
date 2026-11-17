-- AA-R11 (auto-approval Plan 2): a covering photographer's own price in a shared race.
-- NULL price_php = the race price (events.price_per_photo); a set price is between the race
-- price and the ceiling (RacePriceService), once per cooldown (price_changed_at).
ALTER TABLE event_photographer
    ADD COLUMN price_php        NUMERIC(12,2) NULL CHECK (price_php IS NULL OR price_php > 0),
    ADD COLUMN price_changed_at TIMESTAMPTZ   NULL;

-- Every race price change: the photographer's own (SET), an admin reset (RESET, AA-R12) or a
-- clamp when the race price changes (CLAMP, AA-R12).
CREATE TABLE race_price_changes (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id        UUID          NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    photographer_id UUID          NOT NULL REFERENCES users (id),
    changed_by      UUID          NULL REFERENCES users (id) ON DELETE SET NULL,
    from_php        NUMERIC(12,2) NOT NULL,
    to_php          NUMERIC(12,2) NOT NULL,
    reason          VARCHAR(8)    NOT NULL CHECK (reason IN ('SET', 'RESET', 'CLAMP')),
    changed_at      TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX idx_race_price_changes_event ON race_price_changes (event_id, changed_at DESC);
