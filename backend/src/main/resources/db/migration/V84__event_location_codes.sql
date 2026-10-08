ALTER TABLE events
    ADD COLUMN region_code VARCHAR(20),
    ADD COLUMN province_code VARCHAR(40),
    ADD CONSTRAINT ck_events_location_pair CHECK (
        (region_code IS NULL AND province_code IS NULL)
        OR (region_code IS NOT NULL AND province_code IS NOT NULL
            AND region_code ~ '[^[:space:]]' AND province_code ~ '[^[:space:]]')
    );
