-- Operator-only: psql -v event_id=<uuid> -v limit=<row count> [-v apply=true] -f scripts/backfill-photo-metadata.sql
-- Dry-run is the default. Stop between batches to pause seeding; the recovery sweep can be disabled separately.
\set ON_ERROR_STOP on
\if :{?event_id}
\else
  \echo 'event_id is required'
  \set event_id invalid
\endif
\if :{?limit}
\else
  \echo 'limit is required'
  \set limit invalid
\endif
\if :{?apply}
\else
  \set apply false
\endif

\if :apply
WITH candidates AS (
    SELECT p.id FROM photos p
    WHERE p.event_id = :'event_id'::uuid
      AND p.deletion_started_at IS NULL
      AND NOT EXISTS (SELECT 1 FROM photo_metadata m WHERE m.photo_id = p.id)
    ORDER BY p.uploaded_at, p.id
    LIMIT :'limit'::integer
)
INSERT INTO photo_metadata (photo_id, state, extractor_version)
SELECT id, 'PENDING', 1 FROM candidates
ON CONFLICT DO NOTHING;
\else
WITH candidates AS (
    SELECT p.id, p.uploaded_at FROM photos p
    WHERE p.event_id = :'event_id'::uuid
      AND p.deletion_started_at IS NULL
      AND NOT EXISTS (SELECT 1 FROM photo_metadata m WHERE m.photo_id = p.id)
    ORDER BY p.uploaded_at, p.id
    LIMIT :'limit'::integer
)
SELECT id, count(*) OVER () AS candidate_count FROM candidates ORDER BY uploaded_at, id;
\endif
