ALTER TABLE photos
    ADD COLUMN removal_state VARCHAR(10) NOT NULL DEFAULT 'NONE'
        CHECK (removal_state IN ('NONE', 'REVIEW', 'REMOVED')),
    ADD COLUMN purged_at TIMESTAMPTZ;
