-- Independent authorization and resource snapshots survive deletion of a photo.
CREATE TABLE photo_erasure_jobs (
    id uuid PRIMARY KEY, photo_id uuid NOT NULL UNIQUE, event_id uuid NOT NULL, owner_id uuid,
    purpose varchar(24) NOT NULL, authorized_actor_type varchar(16) NOT NULL,
    authorized_actor_id uuid NOT NULL, reason text NOT NULL, request_id uuid, item_id uuid,
    command_id uuid NOT NULL, manifest jsonb NOT NULL,
    additional_intents jsonb NOT NULL DEFAULT '[]',
    authorizations jsonb NOT NULL DEFAULT '[]', retain_tombstone boolean NOT NULL DEFAULT false,
    resource_outcomes jsonb NOT NULL DEFAULT '{}', face_outcome varchar(24) NOT NULL DEFAULT 'PENDING',
    outcome varchar(24) NOT NULL DEFAULT 'PENDING', generation bigint NOT NULL,
    revision bigint NOT NULL DEFAULT 0, attempts integer NOT NULL DEFAULT 0,
    lease_token uuid, lease_until timestamptz, next_attempt_at timestamptz NOT NULL,
    last_error varchar(64), execution_actor_type varchar(16), execution_actor_id uuid,
    created_at timestamptz NOT NULL, updated_at timestamptz NOT NULL, completed_at timestamptz,
    CHECK ((lease_token IS NULL) = (lease_until IS NULL)),
    CHECK (purpose IN ('PHOTOGRAPHER_DELETE', 'PRIVACY_REMOVAL')),
    CHECK (outcome IN ('PENDING', 'RETRY_REQUIRED', 'COMPLETE'))
);
CREATE INDEX photo_erasure_jobs_due ON photo_erasure_jobs(next_attempt_at, id) WHERE outcome <> 'COMPLETE';

ALTER TABLE photo_work_outputs ADD COLUMN cleanup_next_attempt_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP;
CREATE INDEX photo_work_cleanup_due ON photo_work_outputs(cleanup_next_attempt_at) WHERE kind = 'INDEXING' AND faces_erased_at IS NULL;

-- Late callback attribution invalidates confirmation even after the photo is gone.
CREATE FUNCTION reopen_photo_erasure() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' OR NEW.person_ids IS DISTINCT FROM OLD.person_ids
        OR NEW.storage_key IS DISTINCT FROM OLD.storage_key
        OR NEW.job_ids IS DISTINCT FROM OLD.job_ids THEN
        UPDATE photo_erasure_jobs SET outcome = 'PENDING', face_outcome = 'PENDING',
            resource_outcomes = '{}', revision = revision + 1, completed_at = NULL,
            next_attempt_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
        WHERE photo_id = NEW.photo_id;
        UPDATE photos SET purged_at = NULL WHERE id = NEW.photo_id AND purged_at IS NOT NULL;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER photo_erasure_new_output AFTER INSERT OR UPDATE ON photo_work_outputs
    FOR EACH ROW EXECUTE FUNCTION reopen_photo_erasure();
