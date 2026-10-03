-- The direct-upload janitor checks whether any photo still references an
-- intent's original before deleting it. Without this index every cleanup pass
-- scans the whole photos table.
CREATE INDEX idx_photos_s3_key ON photos (s3_key);
