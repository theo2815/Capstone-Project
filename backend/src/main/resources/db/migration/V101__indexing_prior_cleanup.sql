ALTER TABLE photo_work_outputs ADD COLUMN prior_cleanup_finished_at timestamptz;

-- Recording an enrollment result happens after prior cleanup returned, so these old claims have proof.
-- shortcut: legacy claims without a recorded result stay held; recover them only after explicit cleanup reconciliation.
UPDATE photo_work_outputs
SET prior_cleanup_finished_at = CURRENT_TIMESTAMP
WHERE kind = 'INDEXING' AND provider = 'rekognition' AND finished_at IS NULL
  AND batch_id IS NULL AND jsonb_array_length(person_ids) > 0;
