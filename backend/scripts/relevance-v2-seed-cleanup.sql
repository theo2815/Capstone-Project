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
-- Kept: the batch_relevance rows (version 1 uses them too), admin decision records, flags (an ended payout hold
-- stays as a closed flag), the "Payout hold lifted" notes, and users.terms_version (the seed's report lists the
-- values from before; put them back by hand if you need to). Run it at the end of the walk-through, with
-- AI_API_ENABLED still off, so no check starts on a seeded batch meanwhile.
--
-- It prints one count per step.
--
-- Run it on the local Docker database (PowerShell, from the repo root):
--   Get-Content backend/scripts/relevance-v2-seed-cleanup.sql | docker exec -i quickpitik-postgres psql -X -U quickpitik -d quickpitik
-- Run it on staging: railway connect (the staging Postgres), then
-- \i '<repo path with forward slashes>/backend/scripts/relevance-v2-seed-cleanup.sql'. If it stops with an error,
-- type ROLLBACK before anything else.

\set ON_ERROR_STOP on

BEGIN;

CREATE TEMP TABLE seeded_cases ON COMMIT DROP AS
SELECT id, batch_id FROM relevance_cases WHERE trigger = 'SEEDED';

\echo '== Seeded cases found (rows)'
SELECT count(*) AS seeded_cases, count(DISTINCT batch_id) AS seeded_batches FROM seeded_cases;

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

COMMIT;
\echo 'CLEANED. Every SEEDED case, its holds, notes, strikes and alerts, and the scores and words of its batch are gone.'
