-- 2026-05-19: runner orders now mint a share_token too (was guest-only). The
-- new bundle-download endpoint (/api/v1/orders/{id}/download-bundle?token=)
-- is reachable by top-level navigation, which cannot carry the Authorization
-- header — so runners need the same token mechanism guests already use.
-- Backfill existing rows so retroactive bundle downloads work without forcing
-- the runner to place a new order.
--
-- 64 hex chars = two gen_random_uuid() outputs with dashes stripped. No
-- pgcrypto dependency (gen_random_uuid is built-in since Postgres 13). Fits
-- the existing VARCHAR(64) column. UNIQUE constraint on share_token is
-- vanishingly unlikely to collide across 2 * 122 random bits per row.
UPDATE orders
SET share_token =
    replace(gen_random_uuid()::text, '-', '') ||
    replace(gen_random_uuid()::text, '-', '')
WHERE share_token IS NULL;
