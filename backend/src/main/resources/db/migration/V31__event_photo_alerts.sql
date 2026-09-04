-- V31 — "Notify me when my photos are ready" opt-in.
-- One row per (event, runner). notified_at doubles as the single-send
-- idempotency stamp: the date-based sweep (EventPhotosReadySweep) claims it with
-- a conditional UPDATE (EventPhotoAlertRepository.claimNotify) before sending,
-- so a runner is emailed exactly once. No separate notifications table.
--
-- Named event_photo_alerts, not event_registrations, to avoid colliding with
-- the participants/registration table reserved by events.participant_count.
--
-- ON DELETE: event_id / user_id CASCADE (wiping an event or user clears opt-ins).
--            selfie_id SET NULL — a runner may delete the selfie they picked;
--            the notifier then falls back to their primary/latest selfie.

CREATE TABLE event_photo_alerts (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id    UUID        NOT NULL REFERENCES events (id)       ON DELETE CASCADE,
    user_id     UUID        NOT NULL REFERENCES users (id)        ON DELETE CASCADE,
    selfie_id   UUID                 REFERENCES user_selfies (id) ON DELETE SET NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    notified_at TIMESTAMPTZ,
    CONSTRAINT uq_event_photo_alert UNIQUE (event_id, user_id)
);

-- Sweep selection reads un-notified opt-ins; the partial index keeps the scan
-- cheap once most rows are notified (same trick as V23's retryable-photo index).
CREATE INDEX idx_event_photo_alerts_pending
    ON event_photo_alerts (event_id)
    WHERE notified_at IS NULL;
