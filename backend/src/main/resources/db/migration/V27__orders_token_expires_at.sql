-- V27 — Give orders.share_token an expiry.
--
-- share_token authorizes three unauthenticated endpoints:
--   GET /orders/{id}/status?token=
--   GET /orders/{id}?token=          (full detail, incl. clean-original previews)
--   GET /orders/{id}/download-bundle?token=
--
-- It is minted once at create time and never rotated, so until now a token that
-- leaked (forwarded receipt, shared link, mail archive) authorized those reads
-- for as long as the row existed. The download_grants row bounds the *bundle*
-- at paid_at + 1 year, but nothing bounded the token itself.
--
-- 90 days is the window a receipt link stays useful: it covers the "I'll grab
-- these later" case with room to spare, and expires long before the 1-year
-- entitlement. Signed-in runners are unaffected — GET /me/orders/{id} is
-- JWT-gated and never consults the token, so their own re-downloads keep
-- working for the full grant lifetime.
--
-- Backfill uses created_at so existing orders get a window measured from when
-- they were placed, not from when this migration ran. Rows older than 90 days
-- are expired on arrival, which is the intended outcome.
--
-- The authoritative window for NEW orders is app.platform.share-token-ttl —
-- OrderService.create sets the column explicitly. The DB-level DEFAULT exists
-- only so a NOT NULL column can be added to a live table without breaking an
-- older build that is still inserting orders and knows nothing about it.
--
-- Order matters: the DEFAULT is attached AFTER the backfill. Postgres fills
-- existing rows with the default at ADD COLUMN time, so declaring it up front
-- would stamp every historical order with `now() + 90 days` and leave the
-- created_at-relative UPDATE below matching zero rows.

ALTER TABLE orders
    ADD COLUMN token_expires_at TIMESTAMPTZ;

UPDATE orders
SET token_expires_at = created_at + INTERVAL '90 days';

ALTER TABLE orders
    ALTER COLUMN token_expires_at SET DEFAULT (now() + INTERVAL '90 days');

ALTER TABLE orders
    ALTER COLUMN token_expires_at SET NOT NULL;
