-- relevance-v2-seed-cleanup.sql: remove what relevance-v2-seed.sql wrote and what the walk-through then made of it
-- (RV-1.12, RV-4.4; G15).
--
-- Who runs it: the owner, by hand, on STAGING, at the end of the record-only or the auto-hide walk-through.
--
-- It refuses (error, no change) while a SEEDED case still holds the photographer's payouts (a Copied case applied in
-- auto-hide whose hold flag is still open): end that hold first ("End hold" on the case, or restore the case), so
-- the photographer gets the "Payout hold lifted" note and nothing is left holding real payouts.
--
-- What it does, in one transaction, for the batches that have a SEEDED case:
--   flag_hidden_photos: deletes every hold row of those batches (automatic, admin and photographer holds), then puts
--     each photo they held back on sale by the app's one restore rule: only a HIDDEN photo that no other hold row and
--     no hidden photo flag still holds; LIVE, or PROCESSING when it never got a watermark.
--   relevance_strikes: deletes the strikes of the SEEDED cases (confirmed in the walk-through).
--   photographer_messages: deletes the relevance notes of the SEEDED cases (verdict, appeal and blurry notes).
--   then, as before: the alerts of the SEEDED cases (relevance_alerts), the scores (relevance_photo_checks) and words
--   (batch_relevance_words) of those batches, and the SEEDED cases (their sample rows and appeals go with them).
-- Nothing else: no other case, and no row of a batch without a SEEDED case. The seed refuses a batch that already
-- has a case, a score, a word or a hold, so every such row of a seeded batch is the seed's or the walk-through's.
--   users: each photographer whose Terms version the seed stamped (-v terms_version) gets the old version and
--     acceptance date back (NULL stays NULL), unless their version changed since (a real acceptance is kept).
-- Kept: the batch_relevance rows (version 1 uses them too), admin decision records, flags (an ended payout hold
-- stays as a closed flag), the "Payout hold lifted" notes, and every e-mail already sent. Run it at the end of the
-- walk-through with AI_API_ENABLED still off and RELEVANCE_MODE on record-only, so no tick hides, holds or checks a
-- seeded batch meanwhile.
--
-- It prints one count per step.
--
-- It also refuses (RVF-17), before any delete, when a batch with a SEEDED case also holds a case that is not
-- SEEDED: that is a real check, and deleting the batch's scores and words would break it.
--
-- DRY RUN BY DEFAULT (RVF-17). Without -v confirm=yes it does every step, prints the counts, then rolls back and
-- deletes nothing. It first prints where it runs: the database, the server address, the newest migration and the
-- counts of users and photos. SQL cannot tell staging from production: read that line before confirm=yes.
--
-- Run it on the local Docker database (PowerShell, from the repo root):
--   Get-Content backend/scripts/relevance-v2-seed-cleanup.sql | docker exec -i quickpitik-postgres psql -X -U quickpitik -d quickpitik
--   Add  -v confirm=yes  to delete.
-- Run it on staging: railway connect (the staging Postgres), then
-- \i '<repo path with forward slashes>/backend/scripts/relevance-v2-seed-cleanup.sql' (the dry run), then
-- \set confirm yes and \i again. If it stops with an error, type ROLLBACK before anything else.

\set ON_ERROR_STOP on

\if :{?confirm}
  \set qp_confirm :confirm
  \unset confirm
\else
  \set qp_confirm no
\endif

\echo '== Where this runs (SQL cannot tell staging from production: check before confirm=yes)'
SELECT current_database() AS database, inet_server_addr() AS server, inet_server_port() AS port,
       (SELECT version FROM flyway_schema_history WHERE success ORDER BY installed_rank DESC LIMIT 1) AS newest_migration,
       (SELECT count(*) FROM users) AS users, (SELECT count(*) FROM photos) AS photos;

BEGIN;

CREATE TEMP TABLE seeded_cases ON COMMIT DROP AS
SELECT id, batch_id FROM relevance_cases WHERE trigger = 'SEEDED';

\echo '== Seeded cases found (rows)'
SELECT count(*) AS seeded_cases, count(DISTINCT batch_id) AS seeded_batches FROM seeded_cases;

DO $$
DECLARE
    real_batch uuid;
BEGIN
    SELECT c.batch_id INTO real_batch FROM relevance_cases c
    WHERE c.batch_id IN (SELECT batch_id FROM seeded_cases) AND c.trigger <> 'SEEDED'
    ORDER BY c.batch_id LIMIT 1;
    IF real_batch IS NOT NULL THEN
        RAISE EXCEPTION 'Refused: batch % also holds a real relevance case; the clean-up would delete its scores and words. Nothing was deleted.', real_batch;
    END IF;
END
$$;

DO $$
DECLARE
    held text;
BEGIN
    SELECT string_agg(c.id::text, ', ') INTO held
    FROM relevance_cases c JOIN flags f ON f.id = c.hold_flag_id
    WHERE c.id IN (SELECT id FROM seeded_cases) AND f.status IN ('open', 'escalated') AND f.payout_hold_at IS NOT NULL;
    IF held IS NOT NULL THEN
        RAISE EXCEPTION 'Refused: seeded case(s) % still hold payouts. End each hold first (End hold on the case).', held;
    END IF;
END
$$;

\echo '== Deleted: flag_hidden_photos of the seeded batches (the photos they held, kept for the restore below)'
CREATE TEMP TABLE released ON COMMIT DROP AS
WITH gone AS (DELETE FROM flag_hidden_photos WHERE batch_id IN (SELECT batch_id FROM seeded_cases) RETURNING photo_id)
SELECT DISTINCT photo_id FROM gone;
SELECT count(*) AS photos_released FROM released;

\echo '== Back on sale: photos of the seeded batches no other hold or hidden photo flag keeps hidden'
UPDATE photos p
SET status = CASE WHEN p.watermark_s3_key IS NULL THEN 'PROCESSING' ELSE 'LIVE' END, processing_attempts = 0
WHERE p.id IN (SELECT photo_id FROM released) AND p.status = 'HIDDEN'
  AND NOT EXISTS (SELECT 1 FROM flags f WHERE f.target_kind = 'photo' AND f.target_id = p.id AND f.status = 'hidden')
  AND NOT EXISTS (SELECT 1 FROM flag_hidden_photos o WHERE o.photo_id = p.id AND o.restored_at IS NULL);

\echo '== Put back: Terms version and acceptance date of each photographer the seed stamped (if not accepted again since)'
UPDATE users u SET terms_version = s.before, terms_accepted_at = s.accepted_before
FROM (SELECT DISTINCT ON (br.photographer_id) br.photographer_id AS id, j->>'terms_stamped' AS stamped,
             j->>'terms_before' AS before, (j->>'accepted_before')::timestamptz AS accepted_before
      FROM relevance_cases c JOIN batch_relevance br ON br.id = c.batch_id
      -- Only the seed's own JSON is read: other SEEDED cases (step 9's re-checks, older rows) may hold other text.
      CROSS JOIN LATERAL (SELECT CASE WHEN c.summary_sent LIKE '%"terms_stamped"%' THEN c.summary_sent::jsonb END AS j) x
      WHERE c.id IN (SELECT id FROM seeded_cases) AND j ? 'terms_stamped'
      ORDER BY br.photographer_id, c.created_at) s
WHERE u.id = s.id AND u.terms_version = s.stamped;

\echo '== Deleted: relevance_strikes of the seeded cases'
DELETE FROM relevance_strikes WHERE case_id IN (SELECT id FROM seeded_cases);
\echo '== Deleted: photographer_messages (relevance notes) of the seeded cases'
DELETE FROM photographer_messages WHERE relevance_case_id IN (SELECT id FROM seeded_cases);
\echo '== Deleted: relevance_alerts of the seeded cases'
DELETE FROM relevance_alerts WHERE case_id IN (SELECT id FROM seeded_cases);
\echo '== Deleted: relevance_photo_checks of the seeded batches'
DELETE FROM relevance_photo_checks WHERE batch_id IN (SELECT batch_id FROM seeded_cases);
\echo '== Deleted: batch_relevance_words of the seeded batches'
DELETE FROM batch_relevance_words WHERE batch_id IN (SELECT batch_id FROM seeded_cases);
\echo '== Deleted: relevance_cases (SEEDED; their relevance_case_photos and appeals go with them)'
DELETE FROM relevance_cases WHERE id IN (SELECT id FROM seeded_cases);

\if :qp_confirm
COMMIT;
\echo 'CLEANED. Every SEEDED case, its holds, notes, strikes and alerts, and the scores and words of its batch are gone.'
\else
ROLLBACK;
\echo 'DRY RUN: nothing was written. Run again with -v confirm=yes to write.'
\endif
