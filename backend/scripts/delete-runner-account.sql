-- delete-runner-account.sql: delete one RUNNER (buyer) account when the person asks.
--
-- Who runs it: the owner, by hand, once per deletion request. Photographer accounts are
-- not deleted here (their sales ledger would cascade away); admins are not either.
--
-- DRY RUN BY DEFAULT. Without -v confirm=yes it only prints a report and changes nothing.
--
-- What it does, in one transaction:
--   1. Refuses (error, no change) unless user_id is an existing RUNNER that owns no
--      photographer sales, payouts, photos or settings (a role changed by hand).
--   2. Prints a report: the account; rows deleted with it; rows kept with their link to
--      the account cleared, and the personal data that stays on them (money and case
--      records, kept on purpose); the R2 files to delete.
--   3. With confirm=yes: deletes the users row (checking exactly one row went) and commits.
--      Foreign keys do the rest. CASCADE: tokens, saved selfies, saved events, cart, photo
--      alerts, messages. SET NULL: orders, payments stay with their order, ledger rows,
--      disputes, flags, photo removal requests. No AWS call is needed: saved selfies are
--      never indexed at Rekognition.
--
-- Steps for each request:
--   0. Once, before the first real run: check that Railway has a backup of the production
--      database. After COMMIT only a backup can undo this.
--   a. Dry run. Check the account is the requester's; save the output (it lists the R2 keys).
--   b. Run again with confirm=yes.
--   c. R2 (photo bucket quickpitik-prod). From the monorepo checkout:
--        cf auth whoami          -- must show account 277f145551e9715f57bc8e8f70182d5d; stop if not
--        cf r2 objects delete <key> --bucket-name quickpitik-prod --force    -- each saved key
--        cf r2 objects list --bucket-name quickpitik-prod --prefix selfies/<user_id>/
--        cf r2 objects list --bucket-name quickpitik-prod --prefix avatars/<user_id>/
--      Both lists must come back empty; delete anything still there the same way (old
--      avatars no row points at). Without --force, cf prints "Aborted." and still exits 0.
--   d. Reply to the requester, then fill the private request log (Google Drive of
--      quickpitiksupport@gmail.com; never in git or the vault).
--   If the confirmed run fails on uq_orders_guest_idempotency_key_event, a guest order
--   already uses the same email, key and event as one of the runner's orders. Nothing was
--   deleted (it rolled back); ask a developer before going further.
--
-- Run it on the local Docker database (PowerShell, from the repo root):
--   Get-Content backend/scripts/delete-runner-account.sql | docker exec -i quickpitik-postgres psql -X -U quickpitik -d quickpitik -v user_id=<uuid>
--   Add  -v confirm=yes  to delete.
--
-- Run it on production (needs the PostgreSQL 16 command-line tools installed, for psql):
--   In an empty folder:  railway link --project quickpitik-backend --environment production
--                        railway connect            -- pick the Postgres service; opens psql,
--                                                   -- no password is printed
--   Then, inside psql:   \set user_id <uuid>
--                        \i 'C:/full/path/to/backend/scripts/delete-runner-account.sql'
--                        \set user_id <uuid>
--                        \set confirm yes
--                        \i 'C:/full/path/to/backend/scripts/delete-runner-account.sql'
--   Quote the path and use forward slashes (the checkout path has spaces).
--   The script clears user_id and confirm as it starts, so each run must set them again.
--   If it stops with an error inside psql, type ROLLBACK before anything else.

\set ON_ERROR_STOP on

\if :{?user_id}
  \set qp_user_id :user_id
  \unset user_id
\else
  \set qp_user_id ''
\endif
\if :{?confirm}
  \set qp_delete :confirm
  \unset confirm
\else
  \set qp_delete no
\endif

BEGIN;
SELECT set_config('qp.delete_user_id', :'qp_user_id', true) AS user_id;

DO $$
DECLARE
    uid_text text := current_setting('qp.delete_user_id');
    uid uuid;
    user_role text;
BEGIN
    IF uid_text = '' THEN
        RAISE EXCEPTION 'Refused: pass -v user_id=<uuid>';
    END IF;
    uid := uid_text::uuid;
    -- Lock the row so its role cannot change before the delete.
    SELECT role INTO user_role FROM users WHERE id = uid FOR UPDATE;
    IF user_role IS NULL THEN
        RAISE EXCEPTION 'Refused: no user with id %', uid;
    END IF;
    IF user_role <> 'RUNNER' THEN
        RAISE EXCEPTION 'Refused: user % is %; this script deletes RUNNER accounts only', uid, user_role;
    END IF;
    IF EXISTS (SELECT 1 FROM transactions WHERE photographer_id = uid)
        OR EXISTS (SELECT 1 FROM payout_cycles WHERE photographer_id = uid)
        OR EXISTS (SELECT 1 FROM payout_reports WHERE photographer_id = uid)
        OR EXISTS (SELECT 1 FROM photos WHERE photographer_id = uid)
        OR EXISTS (SELECT 1 FROM photographer_settings WHERE user_id = uid)
        OR EXISTS (SELECT 1 FROM photographer_payout_accounts WHERE user_id = uid) THEN
        RAISE EXCEPTION 'Refused: user % is a RUNNER but still owns photographer sales, payouts, photos, settings or payout accounts', uid;
    END IF;
END
$$;

\echo '== Account'
SELECT id, email, name, role, created_at, suspended_at FROM users WHERE id = :'qp_user_id';

\echo '== Deleted with the account (rows)'
SELECT 'refresh_tokens' AS deleted, count(*) AS rows FROM refresh_tokens WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'password_reset_tokens', count(*) FROM password_reset_tokens WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'email_verification_tokens', count(*) FROM email_verification_tokens WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'email_change_tokens', count(*) FROM email_change_tokens WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'desktop_auth_codes', count(*) FROM desktop_auth_codes WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'user_selfies', count(*) FROM user_selfies WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'saved_events', count(*) FROM saved_events WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'cart_items', count(*) FROM cart_items WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'event_photo_alerts', count(*) FROM event_photo_alerts WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'runner_messages', count(*) FROM runner_messages WHERE runner_id = :'qp_user_id';

\echo '== Kept, with the link to the account cleared (rows), and the personal data that stays on them'
SELECT 'orders.user_id' AS link_cleared, count(*) AS rows, 'recipient_email' AS personal_data_kept FROM orders WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'transactions.buyer_id', count(*), 'buyer_display_name' FROM transactions WHERE buyer_id = :'qp_user_id'
UNION ALL SELECT 'disputes.runner_id', count(*), 'note, refund_reason, refund_link_url' FROM disputes WHERE runner_id = :'qp_user_id'
UNION ALL SELECT 'disputes.refund_requested_by', count(*), '' FROM disputes WHERE refund_requested_by = :'qp_user_id'
UNION ALL SELECT 'flags.reporter_id', count(*), 'note' FROM flags WHERE reporter_id = :'qp_user_id'
UNION ALL SELECT 'photo_removal_requests.requester_user_id', count(*), 'requester_contact, requester_reason, evidence file in R2 (kept with the case)' FROM photo_removal_requests WHERE requester_user_id = :'qp_user_id'
UNION ALL SELECT 'admin_decision_log.target_user_id', count(*), 'reason, meta (may name the person)' FROM admin_decision_log WHERE target_user_id = :'qp_user_id';

\echo '== R2 files to delete after the confirmed run (then list selfies/<id>/ and avatars/<id>/: both must be empty)'
SELECT s3_key AS r2_key FROM user_selfies WHERE user_id = :'qp_user_id'
UNION ALL SELECT avatar_s3_key FROM users WHERE id = :'qp_user_id' AND avatar_s3_key IS NOT NULL;

\if :qp_delete
DO $$
DECLARE deleted int;
BEGIN
    DELETE FROM users WHERE id = current_setting('qp.delete_user_id')::uuid AND role = 'RUNNER';
    GET DIAGNOSTICS deleted = ROW_COUNT;
    IF deleted <> 1 THEN
        RAISE EXCEPTION 'Expected to delete 1 user row, deleted %', deleted;
    END IF;
END
$$;
COMMIT;
\echo 'DELETED. Now delete the R2 files listed above and confirm both prefixes are empty.'
\else
ROLLBACK;
\echo 'DRY RUN: nothing changed. Run again with -v confirm=yes to delete.'
\endif
