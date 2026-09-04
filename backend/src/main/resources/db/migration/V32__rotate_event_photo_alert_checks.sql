-- Rotate bounded photos-ready sweeps so unmatched early opt-ins cannot starve
-- later runners. NULL means the alert has never been checked.
ALTER TABLE event_photo_alerts
    ADD COLUMN last_checked_at TIMESTAMPTZ;
