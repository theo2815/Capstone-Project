-- V92: when an event's face templates (its Rekognition collection) were deleted.
-- EventFaceExpiry deletes them 6 months after the event date, or when an admin
-- deletes the event (legal review M2, 2026-11-04). Empty = not deleted yet; a
-- failed delete leaves it empty so the next daily run retries.
ALTER TABLE events
    ADD COLUMN faces_deleted_at TIMESTAMPTZ;
