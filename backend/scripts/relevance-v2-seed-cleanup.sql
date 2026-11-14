-- relevance-v2-seed-cleanup.sql: remove what relevance-v2-seed.sql wrote (RV-1.12, G15).
--
-- Who runs it: the owner, by hand, on STAGING, at the end of the record-only walk-through.
--
-- What it deletes, in one transaction:
--   every relevance_cases row with trigger SEEDED (their sample rows in relevance_case_photos go with them);
--   the alerts of those cases (relevance_alerts);
--   the scores (relevance_photo_checks) and words (batch_relevance_words) of the batches that have a SEEDED case.
-- Nothing else: no other case, and no score or word of a batch without a SEEDED case. The seed refuses a batch
-- that already has a case, a score or a word, so every score and word of a seeded batch is the seed's own.
-- Kept: the batch_relevance rows (version 1 uses them too). Run it with RELEVANCE_MODE back on off, or at
-- least after the walk-through, so no check starts on a seeded batch meanwhile.
--
-- It prints one "DELETE n" count per table.
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

\echo '== Deleted: relevance_alerts of the seeded cases'
DELETE FROM relevance_alerts WHERE case_id IN (SELECT id FROM seeded_cases);
\echo '== Deleted: relevance_photo_checks of the seeded batches'
DELETE FROM relevance_photo_checks WHERE batch_id IN (SELECT batch_id FROM seeded_cases);
\echo '== Deleted: batch_relevance_words of the seeded batches'
DELETE FROM batch_relevance_words WHERE batch_id IN (SELECT batch_id FROM seeded_cases);
\echo '== Deleted: relevance_cases (SEEDED; their relevance_case_photos go with them)'
DELETE FROM relevance_cases WHERE id IN (SELECT id FROM seeded_cases);

COMMIT;
\echo 'CLEANED. Every SEEDED case and the scores and words of its batch are gone.'
