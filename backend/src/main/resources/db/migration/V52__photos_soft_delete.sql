-- V52: photographer photo soft-delete (trash) — 2026-09-07
--   Photographers can move their own event photos to a recoverable "Deleted
--   Photos" area, restore them, or permanently delete them. Trash is a
--   soft-delete timestamp, mirroring events.deleted_at — deliberately NOT a new
--   PhotoStatus value, so a photo's lifecycle status (PROCESSING/LIVE/HIDDEN)
--   survives a trash → restore round-trip unchanged.
--
--   deleted_at : NULL = active (the default, so every existing row stays active
--                and visible — behaviour-preserving). Non-NULL = in the
--                photographer's trash: excluded from every runner / public /
--                search / checkout surface and shown only in Deleted Photos.
ALTER TABLE photos
    ADD COLUMN deleted_at TIMESTAMPTZ;

-- The trash view: one photographer's deleted photos within an event. A small
-- set, and the index is partial so it adds nothing to the hot active-gallery
-- path (where deleted_at IS NULL is the overwhelming majority).
CREATE INDEX idx_photos_trash
    ON photos (event_id, photographer_id)
    WHERE deleted_at IS NOT NULL;
