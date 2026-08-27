-- V35: index tuning (2026-08-27 performance pass).

-- 1) The event-grid ORDER BY is `captured_at DESC NULLS LAST, uploaded_at
--    DESC, id ASC`; the V3 index was (event_id, captured_at DESC, id) — in
--    Postgres DESC implies NULLS FIRST, and the tiebreakers don't match, so
--    every grid page forced a materialise + sort. Recreate to mirror the
--    ORDER BY exactly so the planner can walk the index.
DROP INDEX IF EXISTS idx_photos_event_captured_at;
CREATE INDEX idx_photos_event_captured_at
    ON photos (event_id, captured_at DESC NULLS LAST, uploaded_at DESC, id ASC);

-- 2) Six indexes that duplicate a UNIQUE constraint on the same column(s):
--    pure write amplification, zero read benefit — the unique index already
--    serves every lookup the duplicate could.
DROP INDEX IF EXISTS idx_users_email;                        -- dupes users_email_key (V1)
DROP INDEX IF EXISTS idx_refresh_tokens_token_hash;          -- dupes refresh_tokens_token_hash_key (V1)
DROP INDEX IF EXISTS idx_password_reset_tokens_token_hash;   -- dupes password_reset_tokens_token_hash_key (V1)
DROP INDEX IF EXISTS idx_photographer_settings_handle;       -- dupes photographer_settings_handle_key (V7)
DROP INDEX IF EXISTS idx_email_change_tokens_token_hash;     -- dupes email_change_tokens_token_hash_key (V28)
DROP INDEX IF EXISTS idx_email_verification_tokens_token_hash; -- dupes email_verification_tokens_token_hash_key (V30)
