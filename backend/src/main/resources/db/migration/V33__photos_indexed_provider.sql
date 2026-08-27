-- V33: which AI provider produced a photo's stored face/bib results.
-- ai-api person ids are opaque UUIDs; Rekognition ids are "{eventId}.{photoId}"
-- composites — the two ID spaces are incompatible, so after an app.ai.provider
-- flip the stale rows are detectable (and re-drivable via the admin
-- POST /admin/events/{id}/photos/reindex?all=true endpoint) instead of silently
-- unsearchable. Nullable: rows indexed before V33 (or never indexed) carry no stamp.
ALTER TABLE photos
    ADD COLUMN indexed_provider VARCHAR(16);
