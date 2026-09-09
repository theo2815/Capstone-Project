ALTER TABLE photos ADD COLUMN deletion_started_at TIMESTAMPTZ;
ALTER TABLE photos ADD CONSTRAINT ck_photo_deletion_requires_trash
    CHECK (deletion_started_at IS NULL OR deleted_at IS NOT NULL);

-- Repair counts left behind by the original trash implementation.
UPDATE event_photographer ep SET photo_count = (
    SELECT COUNT(*) FROM photos p
    WHERE p.event_id = ep.event_id AND p.photographer_id = ep.photographer_id
      AND p.deleted_at IS NULL
);
UPDATE events e SET photo_count = (
    SELECT COUNT(*) FROM photos p WHERE p.event_id = e.id AND p.deleted_at IS NULL
);
