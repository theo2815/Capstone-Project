-- nuke-non-admin-users.sql
--
-- Wipe every non-admin user + all rows they own, leaving the admin account(s),
-- reference data (events, event_categories, regions, platform fees, flags,
-- reserved_handles), and the database schema itself intact. Idempotent —
-- running twice is a no-op the second time.
--
-- Why this exists: the FE's mock Google OAuth (disabled 2026-05-18 via
-- OAUTH_ENABLED=false in website/src/components/auth/google-button.tsx)
-- hardcoded `juan.delacruz@gmail.com` as the identity for every Google
-- signup, so test accounts appeared to share a profile. This script clears
-- the resulting test data + any half-baked email/password registrations,
-- giving a clean slate to retest with real /auth/register signups.
--
-- HOW TO RUN (PowerShell, from repo root — one line):
--
--   Get-Content backend/scripts/nuke-non-admin-users.sql | docker compose -f backend/docker-compose.yml exec -T postgres psql -U quickpitik -d quickpitik
--
-- Why not `< file`: PowerShell does not have bash-style input redirection.
-- The `<` operator is reserved/unsupported in PS5.1+ and the shell silently
-- glues the filename onto the previous argument, producing errors like
-- `database "quickpitik<backend/..." does not exist`. Pipe Get-Content
-- through docker's stdin instead — works in every PS version.
--
-- ALTERNATIVELY (bash / Git Bash / WSL — one line):
--
--   docker compose -f backend/docker-compose.yml exec -T postgres psql -U quickpitik -d quickpitik < backend/scripts/nuke-non-admin-users.sql
--
-- NOTE: the local-fs storage directory (.storage/ under backend/, or whatever
-- STORAGE_LOCAL_ROOT points to) still holds the uploaded image files. The DB
-- rows are gone so nothing references them — they're orphan bytes. Either:
--   - leave them (harmless, will be overwritten on next upload to same key)
--   - delete the directory manually: `Remove-Item -Recurse -Force backend/.storage`

BEGIN;

-- Defensive guard: refuse to run if no admin exists. Prevents accidentally
-- locking ourselves out of the system entirely.
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM users WHERE role = 'ADMIN') = 0 THEN
        RAISE EXCEPTION 'Refusing to nuke: no ADMIN user exists. Aborting before any DELETE runs.';
    END IF;
END $$;

-- Phase 1: wipe rows whose FK to users is ON DELETE SET NULL.
-- If we skipped this, those rows would survive as orphans after Phase 2.
-- Order matters: child tables first so cascade chains stay clean.

DELETE FROM disputes;            -- CASCADEs from orders + photos, but explicit
DELETE FROM admin_decision_log;  -- runner_id / target_user_id SET NULL on user
DELETE FROM download_grants;     -- CASCADEs from orders, but explicit
DELETE FROM payments;            -- CASCADEs from orders, but explicit
DELETE FROM order_items;         -- CASCADEs from orders, but explicit
DELETE FROM orders;              -- user_id SET NULL on user
DELETE FROM transactions;        -- buyer_id SET NULL on user (V9 earnings)
DELETE FROM photo_face_persons;  -- CASCADEs from photos, but explicit
DELETE FROM photo_bibs;          -- CASCADEs from photos, but explicit
DELETE FROM photos;              -- photographer_id SET NULL on user

-- Phase 2: delete the non-admin users. PostgreSQL CASCADE FKs handle:
--   refresh_tokens, password_reset_tokens (V1)
--   saved_events, cart_items                (V4)
--   user_selfies                            (V6)
--   photographer_settings, event_photographer (V7)
--   social_links, photographer_payout_accounts (V8)
--   payout_cycles, payout_reports           (V9)
--   photographer_messages                   (V10)

DELETE FROM users WHERE role <> 'ADMIN';

-- Phase 3: verification. Visible in the transaction before COMMIT so a
-- mistaken run can be rolled back. Counts should all be 0 except admins.
SELECT 'After nuke' AS phase,
       (SELECT COUNT(*) FROM users)                    AS users_total,
       (SELECT COUNT(*) FROM users WHERE role = 'ADMIN') AS admins,
       (SELECT COUNT(*) FROM users WHERE role <> 'ADMIN') AS non_admins,
       (SELECT COUNT(*) FROM photos)                   AS photos,
       (SELECT COUNT(*) FROM orders)                   AS orders,
       (SELECT COUNT(*) FROM photographer_settings)    AS photographer_settings,
       (SELECT COUNT(*) FROM user_selfies)             AS selfies,
       (SELECT COUNT(*) FROM cart_items)               AS cart_items,
       (SELECT COUNT(*) FROM saved_events)             AS saved_events,
       (SELECT COUNT(*) FROM refresh_tokens)           AS refresh_tokens,
       (SELECT COUNT(*) FROM photographer_messages)    AS photographer_messages;

COMMIT;
