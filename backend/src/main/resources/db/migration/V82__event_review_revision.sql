ALTER TABLE events ADD COLUMN review_revision BIGINT NOT NULL DEFAULT 0;
ALTER TABLE events ADD CONSTRAINT ck_events_review_revision_nonnegative CHECK (review_revision >= 0);
