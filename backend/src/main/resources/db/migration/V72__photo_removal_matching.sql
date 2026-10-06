-- Latest discovery is separate from immutable selected-item provenance.
ALTER TABLE photo_removal_requests ADD COLUMN candidate_cache JSONB
    CHECK (candidate_cache IS NULL OR jsonb_typeof(candidate_cache) = 'object');

ALTER TABLE photo_removal_commands DROP CONSTRAINT photo_removal_commands_operation_check;
ALTER TABLE photo_removal_commands ADD CONSTRAINT photo_removal_commands_operation_check
    CHECK (operation IN ('CREATE', 'REPLACE_EVIDENCE', 'CANCEL', 'MATCH_CANDIDATES'));
