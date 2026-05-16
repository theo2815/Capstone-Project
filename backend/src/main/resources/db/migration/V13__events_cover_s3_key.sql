-- V13: events.cover_s3_key for admin-uploaded cover banners.
--
-- Problem: POST /api/v1/admin/events historically accepted `bannerUrl` as a
-- JSON string and stored it in events.banner_url VARCHAR(512). The FE form
-- (admin-event-form-modal.tsx) produced a base64 data URL from the picked
-- file via fitToDataUrl(file, 1920, 0.82) — a single cover image was 30-100
-- KB of base64, blowing past VARCHAR(512) and yielding a 500 on every
-- create-with-cover (and forcing covers out of /admin/events entirely).
--
-- Fix: cover binaries belong in object storage (LocalFsStorageService in
-- dev, S3 in prod), with the row pointing at the storage key. Same pattern
-- as users.avatar_s3_key (PR 6). The FE switches to multipart so the bytes
-- never round-trip through the JSON layer.
--
-- Compatibility: events.banner_url stays for V2 seed rows that ship pre-
-- populated URLs and for any caller that ever set a remote URL directly.
-- The mapper resolves cover_s3_key first (presigned GET, app.storage.cover
-- TTL) and falls back to banner_url when the key is null.
--
-- Backfill: not needed. Existing rows have NULL cover_s3_key, mapper falls
-- back to banner_url (which is NULL for V2 seeds — they render the
-- text-only banner fallback the FE already paints).

ALTER TABLE events
    ADD COLUMN cover_s3_key VARCHAR(512);
