ALTER TABLE photos
    ADD COLUMN work_generation bigint NOT NULL DEFAULT 0,
    ADD COLUMN watermark_work_token uuid,
    ADD COLUMN watermark_work_expires_at timestamptz,
    ADD COLUMN indexing_work_token uuid,
    ADD COLUMN indexing_work_expires_at timestamptz,
    ADD CONSTRAINT photo_watermark_work_pair CHECK ((watermark_work_token IS NULL) = (watermark_work_expires_at IS NULL)),
    ADD CONSTRAINT photo_indexing_work_pair CHECK ((indexing_work_token IS NULL) = (indexing_work_expires_at IS NULL));

-- Immutable attribution survives a missing photo. Expiry is not remote completion.
-- TASK 9/10 reconcile unfinished rows and consume all known output targets.
CREATE TABLE photo_work_outputs (
    token uuid PRIMARY KEY,
    photo_id uuid NOT NULL,
    event_id uuid NOT NULL,
    generation bigint NOT NULL,
    kind varchar(12) NOT NULL CHECK (kind IN ('WATERMARK', 'INDEXING')),
    provider varchar(16) NOT NULL,
    expires_at timestamptz NOT NULL,
    storage_key text,
    person_name text,
    person_ids jsonb NOT NULL DEFAULT '[]',
    prior_provider varchar(16),
    prior_person_ids jsonb NOT NULL DEFAULT '[]',
    batch_id uuid,
    job_ids jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at timestamptz
);
CREATE INDEX photo_work_outputs_photo ON photo_work_outputs(photo_id);
CREATE INDEX photo_work_outputs_batch ON photo_work_outputs(batch_id) WHERE batch_id IS NOT NULL;

-- Jobs submitted before this migration already exist remotely. An expired local
-- lease deliberately prevents publishing; it does not declare their writes over.
INSERT INTO photo_work_outputs(token, photo_id, event_id, generation, kind, provider,
    expires_at, person_name, batch_id, job_ids)
SELECT gen_random_uuid(), bp.photo_id, b.event_id, COALESCE(p.work_generation, 0), 'INDEXING', 'ai_api',
    b.created_at, bp.photo_id::text, b.id,
    COALESCE((SELECT jsonb_object_agg(j.kind, j.ai_job_id) FROM ai_index_jobs j WHERE j.batch_id = b.id), '{}')
FROM ai_index_batches b JOIN ai_index_batch_photos bp ON bp.batch_id = b.id
LEFT JOIN photos p ON p.id = bp.photo_id
WHERE b.status = 'SUBMITTED';

UPDATE photos p SET indexing_work_token = w.token, indexing_work_expires_at = w.expires_at
FROM (SELECT DISTINCT ON (photo_id) photo_id, token, expires_at
      FROM photo_work_outputs WHERE kind = 'INDEXING' ORDER BY photo_id, expires_at DESC, token) w
WHERE p.id = w.photo_id;
