-- AX-60: when a rejected event was sent again for review. NULL for a first submission
-- and for every event resent before this migration. Additive only.
ALTER TABLE events ADD COLUMN resubmitted_at TIMESTAMPTZ;
