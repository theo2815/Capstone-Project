-- V24: per-photographer content-hash duplicate detection
--
-- A photo's identity is the SHA-256 hexdigest of its ORIGINAL uploaded bytes
-- (computed by PhotoUploadService BEFORE watermarking, so client- and
-- server-side hashes of the same file agree). content_hash lets the backend
-- detect when the same shot is uploaded twice across BOTH upload doors — mobile
-- camera-card import / tether AND the website — which is the only place the two
-- clients converge.
--
-- Boundary is per photographer across all events: a photographer cannot upload
-- the same image content into any of their events twice. A same-event re-upload
-- returns the existing photo idempotently; a different-event re-upload is
-- rejected by the service.
--
-- Existing rows predate this and stay NULL; the partial unique index excludes
-- NULLs (and ON DELETE SET NULL photographer orphans) so no violation occurs on
-- legacy data. Only new uploads with a computed hash are constrained.

ALTER TABLE photos
    ADD COLUMN content_hash VARCHAR(64);

CREATE UNIQUE INDEX uq_photos_photographer_content_hash
    ON photos (photographer_id, content_hash)
    WHERE photographer_id IS NOT NULL AND content_hash IS NOT NULL;
