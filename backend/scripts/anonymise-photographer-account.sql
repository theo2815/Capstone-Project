-- anonymise-photographer-account.sql: close one PHOTOGRAPHER account when the person asks.
--
-- Who runs it: the owner, by hand, once per deletion request. Runner accounts use
-- delete-runner-account.sql instead.
--
-- Why not delete the row: transactions and payout_cycles cascade on the user, so deleting
-- it would erase the sales ledger and payout history. Instead the account is scrubbed and
-- suspended (owner decision D19), and the money records stay.
--
-- DRY RUN BY DEFAULT. Without -v confirm=yes it only prints a report and changes nothing.
--
-- Before the run, the photographer must have:
--   * deleted every unsold photo in the app (Trash, then Delete forever: that erases the
--     files and the faces). Sold photos stay, because buyers keep their downloads.
--   * been paid everything they are owed, even below the 500 peso minimum, with no payout
--     still open (D20). Money still inside the 7-day clearing window counts as owed.
-- The script refuses (error, no change) until both are true, while an unpaid order still
-- holds one of their photos (a late payment would add a sale), and while the id is not an
-- existing PHOTOGRAPHER. Read the counts in the error to tell the photographer what is left.
--
-- What the confirmed run does, in one transaction:
--   users: name "Deleted photographer", email deleted-<id>@deleted.invalid, a random
--     password nobody knows (bcrypt, like Google accounts get), Google link, avatar and
--     avatar file key cleared, credential_version + 1 (old sign-ins stop working),
--     suspended now with reason "account deleted on request" (the public page then 404s).
--   Deleted: sign-in, reset, email-change and desktop tokens; social links; payout accounts
--     and their audit rows; photographer messages; leftover upload intents; saved selfies,
--     saved events, cart and photo alerts (if they used runner features); live-selfie
--     sessions (liveness_sessions); photo relevance checks (batch_relevance, with their
--     looked-up photo rows).
--   photographer_settings: handle becomes deleted-<id without dashes> (40 characters, longer
--     than the app lets anyone choose); brand name, bio, place, cover and watermark cleared.
--   Their remaining (sold) photos go to the bin, so nobody can buy them any more; buyers keep
--     their downloads, as when a photographer bins a photo in the app.
--   Coupons are switched off, not deleted: past orders keep the link to the coupon they used.
--   Kept: sold photos, orders, the ledger (transactions), payout cycles with their account
--     snapshot (name, number, QR file), payout reports, admin decision records, the name on
--     photo removal cases, orders they placed as a buyer, and the seller-details record (with
--     its ID-number fingerprint) and its files under seller-details/<user_id>/, including the
--     live-selfie reference image (legal records under RA 11967 and BIR rules;
--     see the privacy policy's retention table).
--
-- Steps for each request:
--   0. Once, before the first real run: check that Railway has a backup of the production
--      database. After COMMIT only a backup can undo this.
--   a. Dry run. Check the account is the requester's; save the output (it lists the R2 keys).
--      If it is refused, tell the photographer what is left (photos to delete, money to pay
--      out by hand), then start again.
--   b. Run again with confirm=yes.
--   b2. Wait 15 minutes (a sign-in made before the run lasts that long), then do the dry run
--      again: every "Deleted" count must be 0. If not, run with confirm=yes once more.
--   c. R2 (photo bucket quickpitik-prod). From the monorepo checkout:
--        cf auth whoami          -- must show account 277f145551e9715f57bc8e8f70182d5d; stop if not
--        cf r2 objects delete <key> --bucket-name quickpitik-prod --force    -- each saved key
--      For the staging/photo-uploads/<id>/ line, list that prefix and delete what is there:
--        cf r2 objects list --bucket-name quickpitik-prod --prefix staging/photo-uploads/<user_id>/
--      Then list photographers/<user_id>/ and avatars/<user_id>/ again. Only the payout QR
--      files named under "Kept" may remain; delete anything else the same way.
--      Without --force, cf prints "Aborted." and still exits 0.
--   d. Reply to the requester, then fill the private request log (Google Drive of
--      quickpitiksupport@gmail.com; never in git or the vault).
--
-- Run it on the local Docker database (PowerShell, from the repo root):
--   Get-Content backend/scripts/anonymise-photographer-account.sql | docker exec -i quickpitik-postgres psql -X -U quickpitik -d quickpitik -v user_id=<uuid>
--   Add  -v confirm=yes  to anonymise.
--
-- Run it on production: as in delete-runner-account.sql (railway connect, then \set user_id,
-- \i with the quoted forward-slash path, then \set confirm yes and \i again). If it stops
-- with an error inside psql, type ROLLBACK before anything else.

\set ON_ERROR_STOP on

\if :{?user_id}
  \set qp_user_id :user_id
  \unset user_id
\else
  \set qp_user_id ''
\endif
\if :{?confirm}
  \set qp_anonymise :confirm
  \unset confirm
\else
  \set qp_anonymise no
\endif

BEGIN;
SELECT set_config('qp.anonymise_user_id', :'qp_user_id', true) AS user_id;

DO $$
DECLARE
    uid_text text := current_setting('qp.anonymise_user_id');
    uid uuid;
    user_role text;
    unsold int;
    open_payouts int;
    unpaid_orders int;
    owed numeric;
BEGIN
    IF uid_text = '' THEN
        RAISE EXCEPTION 'Refused: pass -v user_id=<uuid>';
    END IF;
    uid := uid_text::uuid;
    -- Lock the row so its role cannot change, and no payout starts, before the update.
    SELECT role INTO user_role FROM users WHERE id = uid FOR UPDATE;
    IF user_role IS NULL THEN
        RAISE EXCEPTION 'Refused: no user with id %', uid;
    END IF;
    IF user_role <> 'PHOTOGRAPHER' THEN
        RAISE EXCEPTION 'Refused: user % is %; this script closes PHOTOGRAPHER accounts only', uid, user_role;
    END IF;
    -- "Sold" as PhotographerPhotoService.isSold: a download grant, or a paid order line.
    SELECT count(*) INTO unsold FROM photos p
    WHERE p.photographer_id = uid AND p.purged_at IS NULL
      AND NOT EXISTS (SELECT 1 FROM download_grants g WHERE g.photo_id = p.id)
      AND NOT EXISTS (SELECT 1 FROM order_items oi JOIN orders o ON o.id = oi.order_id
                      WHERE oi.photo_id = p.id
                        AND (o.paid_at IS NOT NULL OR o.status IN ('PAID', 'FULFILLED', 'REFUNDED')));
    -- Open = PayoutRequestService.OPEN_STATUS_WIRES.
    SELECT count(*) INTO open_payouts FROM payout_cycles
    WHERE photographer_id = uid AND status IN ('scheduled', 'pending', 'approved', 'processing', 'held');
    SELECT count(DISTINCT o.id) INTO unpaid_orders FROM orders o
    JOIN order_items oi ON oi.order_id = o.id JOIN photos p ON p.id = oi.photo_id
    WHERE p.photographer_id = uid AND o.status = 'PENDING' AND o.paid_at IS NULL;
    SELECT (SELECT coalesce(sum(amount_kept_php), 0) FROM transactions WHERE photographer_id = uid)
         - (SELECT coalesce(sum(amount_php), 0) FROM payout_cycles WHERE photographer_id = uid AND status = 'paid')
      INTO owed;
    IF unsold > 0 OR open_payouts > 0 OR owed > 0 OR unpaid_orders > 0 THEN
        RAISE EXCEPTION 'Refused: photographer % still has % unsold photos not deleted forever, % open payouts, % pesos not paid out, and % unpaid orders holding their photos',
            uid, unsold, open_payouts, owed, unpaid_orders;
    END IF;
END
$$;

\echo '== Account'
SELECT u.id, u.email, u.name, u.role, u.created_at, u.suspended_at, s.handle, s.verification_status
FROM users u LEFT JOIN photographer_settings s ON s.user_id = u.id WHERE u.id = :'qp_user_id';

\echo '== Deleted (rows)'
SELECT 'refresh_tokens' AS deleted, count(*) AS rows FROM refresh_tokens WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'password_reset_tokens', count(*) FROM password_reset_tokens WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'email_verification_tokens', count(*) FROM email_verification_tokens WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'email_change_tokens', count(*) FROM email_change_tokens WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'desktop_auth_codes', count(*) FROM desktop_auth_codes WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'social_links', count(*) FROM social_links WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'photographer_payout_accounts', count(*) FROM photographer_payout_accounts WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'payout_account_audit', count(*) FROM payout_account_audit WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'photographer_messages', count(*) FROM photographer_messages WHERE photographer_id = :'qp_user_id'
UNION ALL SELECT 'photo_upload_intents', count(*) FROM photo_upload_intents WHERE photographer_id = :'qp_user_id'
UNION ALL SELECT 'user_selfies', count(*) FROM user_selfies WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'saved_events', count(*) FROM saved_events WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'cart_items', count(*) FROM cart_items WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'event_photo_alerts', count(*) FROM event_photo_alerts WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'liveness_sessions', count(*) FROM liveness_sessions WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'batch_relevance', count(*) FROM batch_relevance WHERE photographer_id = :'qp_user_id';

\echo '== Kept (rows), and the personal data that stays on them'
SELECT 'photos (sold; binned by the run)' AS kept, count(*) AS rows, 'credit in already-made previews' AS personal_data_kept FROM photos WHERE photographer_id = :'qp_user_id' AND purged_at IS NULL
UNION ALL SELECT 'transactions', count(*), '' FROM transactions WHERE photographer_id = :'qp_user_id'
UNION ALL SELECT 'payout_cycles', count(*), 'account_snapshot_name, account_snapshot_number, account_snapshot_qr_key file' FROM payout_cycles WHERE photographer_id = :'qp_user_id'
UNION ALL SELECT 'payout_reports', count(*), 'note' FROM payout_reports WHERE photographer_id = :'qp_user_id'
UNION ALL SELECT 'photographer_coupons (switched off)', count(*), '' FROM photographer_coupons WHERE photographer_id = :'qp_user_id'
UNION ALL SELECT 'event_photographer', count(*), '' FROM event_photographer WHERE photographer_id = :'qp_user_id'
UNION ALL SELECT 'events created by them', count(*), 'event details they typed' FROM events WHERE created_by = :'qp_user_id'
UNION ALL SELECT 'admin_decision_log.target_user_id', count(*), 'reason, meta (may name the person)' FROM admin_decision_log WHERE target_user_id = :'qp_user_id'
UNION ALL SELECT 'photo_removal_items', count(*), 'owner_name_snapshot (locked by a trigger)' FROM photo_removal_items WHERE owner_id_snapshot = :'qp_user_id'
UNION ALL SELECT 'orders placed as a buyer', count(*), 'recipient_email' FROM orders WHERE user_id = :'qp_user_id'
UNION ALL SELECT 'photographer_seller_details', count(*), 'legal name, address, phone, ID photos, live-selfie reference image, ID-number fingerprint' FROM photographer_seller_details WHERE user_id = :'qp_user_id';

\echo '== R2 files to delete after the confirmed run (a key ending in / is a prefix: list it and delete what is there)'
SELECT cover_s3_key AS r2_key FROM photographer_settings WHERE user_id = :'qp_user_id' AND cover_s3_key IS NOT NULL
UNION ALL SELECT watermark_s3_key FROM photographer_settings WHERE user_id = :'qp_user_id' AND watermark_s3_key IS NOT NULL
UNION ALL SELECT avatar_s3_key FROM users WHERE id = :'qp_user_id' AND avatar_s3_key IS NOT NULL
UNION ALL SELECT s3_key FROM user_selfies WHERE user_id = :'qp_user_id'
UNION ALL SELECT a.qr_s3_key FROM photographer_payout_accounts a
    WHERE a.user_id = :'qp_user_id' AND a.qr_s3_key IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM payout_cycles c WHERE c.account_snapshot_qr_key = a.qr_s3_key)
UNION ALL SELECT i.original_key FROM photo_upload_intents i
    WHERE i.photographer_id = :'qp_user_id' AND i.state <> 'COMMITTED' AND i.cleaned_at IS NULL
      AND NOT EXISTS (SELECT 1 FROM photos p WHERE p.s3_key = i.original_key)
UNION ALL SELECT 'staging/photo-uploads/' || :'qp_user_id' || '/';

\echo '== Kept in R2: payout QR files named by a payout snapshot, and the seller-details files incl. the live-selfie reference image (do not delete)'
SELECT DISTINCT account_snapshot_qr_key AS kept_r2_key FROM payout_cycles
WHERE photographer_id = :'qp_user_id' AND account_snapshot_qr_key IS NOT NULL
UNION ALL SELECT 'seller-details/' || :'qp_user_id' || '/'
WHERE EXISTS (SELECT 1 FROM photographer_seller_details WHERE user_id = :'qp_user_id');

\if :qp_anonymise
DO $$
DECLARE
    uid uuid := current_setting('qp.anonymise_user_id')::uuid;
    changed int;
BEGIN
    UPDATE users SET
        name = 'Deleted photographer',
        email = 'deleted-' || uid || '@deleted.invalid',
        password_hash = crypt(encode(gen_random_bytes(32), 'hex'), gen_salt('bf', 12)),
        google_sub = NULL, avatar_url = NULL, avatar_s3_key = NULL,
        credential_version = credential_version + 1,
        suspended_at = coalesce(suspended_at, now()),
        suspension_reason = 'account deleted on request',
        updated_at = now()
    WHERE id = uid AND role = 'PHOTOGRAPHER';
    GET DIAGNOSTICS changed = ROW_COUNT;
    IF changed <> 1 THEN
        RAISE EXCEPTION 'Expected to anonymise 1 user row, changed %', changed;
    END IF;

    DELETE FROM refresh_tokens WHERE user_id = uid;
    DELETE FROM password_reset_tokens WHERE user_id = uid;
    DELETE FROM email_verification_tokens WHERE user_id = uid;
    DELETE FROM email_change_tokens WHERE user_id = uid;
    DELETE FROM desktop_auth_codes WHERE user_id = uid;
    DELETE FROM social_links WHERE user_id = uid;
    DELETE FROM photographer_payout_accounts WHERE user_id = uid;
    DELETE FROM payout_account_audit WHERE user_id = uid;
    DELETE FROM photographer_messages WHERE photographer_id = uid;
    DELETE FROM photo_upload_intents WHERE photographer_id = uid;
    DELETE FROM event_photo_alerts WHERE user_id = uid;
    DELETE FROM user_selfies WHERE user_id = uid;
    DELETE FROM saved_events WHERE user_id = uid;
    DELETE FROM cart_items WHERE user_id = uid;
    DELETE FROM liveness_sessions WHERE user_id = uid;
    -- A deadlock error here means the relevance sweep was scoring this account: ROLLBACK and run again.
    DELETE FROM batch_relevance WHERE photographer_id = uid;
    UPDATE photographer_coupons SET active = false, updated_at = now() WHERE photographer_id = uid;
    -- Bin what is left (only sold photos pass the checks) and lower the counts, as trash() does.
    WITH binned AS (
        UPDATE photos SET deleted_at = now() WHERE photographer_id = uid AND deleted_at IS NULL RETURNING event_id
    ), per_event AS (
        SELECT event_id, count(*) AS n FROM binned GROUP BY event_id
    ), studio AS (
        UPDATE event_photographer ep SET photo_count = GREATEST(0, ep.photo_count - pe.n)
        FROM per_event pe WHERE ep.event_id = pe.event_id AND ep.photographer_id = uid
    )
    UPDATE events e SET photo_count = GREATEST(0, e.photo_count - pe.n) FROM per_event pe WHERE e.id = pe.event_id;
    UPDATE photographer_settings SET
        handle = 'deleted-' || replace(uid::text, '-', ''),
        brand_name = NULL, bio = '', region_code = NULL, province_code = NULL, city = NULL,
        cover_s3_key = NULL, watermark_s3_key = NULL, watermark_label = NULL,
        updated_at = now()
    WHERE user_id = uid;
END
$$;
COMMIT;
\echo 'ANONYMISED AND SUSPENDED. Now delete the R2 files listed above and check the prefixes.'
\else
ROLLBACK;
\echo 'DRY RUN: nothing changed. Run again with -v confirm=yes to anonymise.'
\endif
