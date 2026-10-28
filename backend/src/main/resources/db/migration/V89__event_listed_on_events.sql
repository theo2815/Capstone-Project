-- 2026-10-28: an admin can show one photographer-owned PUBLIC event on the
-- public /events list. Admin events (created_by IS NULL) are always listed
-- and ignore this column.
ALTER TABLE events ADD COLUMN listed_on_events BOOLEAN NOT NULL DEFAULT FALSE;
