-- Receipts preserve full-command replay and distinguish unsettled storage writes.
CREATE TABLE photo_removal_commands (
    command_id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES photo_removal_requests(id),
    operation VARCHAR(24) NOT NULL CHECK (operation IN ('CREATE', 'REPLACE_EVIDENCE', 'CANCEL')),
    actor_id_snapshot UUID NOT NULL,
    input_hash VARCHAR(64) NOT NULL,
    state VARCHAR(16) NOT NULL CHECK (state IN ('PENDING', 'UNKNOWN', 'SUCCEEDED', 'FAILED')),
    intended_evidence_key VARCHAR(512),
    result JSONB,
    claimed_at TIMESTAMPTZ NOT NULL,
    CHECK (state <> 'SUCCEEDED' OR result IS NOT NULL),
    CHECK (operation = 'REPLACE_EVIDENCE' OR intended_evidence_key IS NULL)
);
CREATE INDEX idx_photo_removal_commands_unsettled ON photo_removal_commands(request_id, state)
    WHERE state IN ('PENDING', 'UNKNOWN');

ALTER TABLE photo_removal_requests ADD COLUMN evidence_cleanup_next_attempt_at TIMESTAMPTZ;
CREATE INDEX idx_photo_removal_evidence_cleanup_due ON photo_removal_requests(evidence_cleanup_next_attempt_at, id)
    WHERE evidence_cleanup_next_attempt_at IS NOT NULL;

CREATE FUNCTION protect_photo_removal_command_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.command_id, NEW.request_id, NEW.operation, NEW.actor_id_snapshot, NEW.input_hash,
           NEW.intended_evidence_key, NEW.claimed_at)
        IS DISTINCT FROM ROW(OLD.command_id, OLD.request_id, OLD.operation, OLD.actor_id_snapshot,
           OLD.input_hash, OLD.intended_evidence_key, OLD.claimed_at)
        OR (OLD.state = 'SUCCEEDED' AND ROW(NEW.state, NEW.result) IS DISTINCT FROM ROW(OLD.state, OLD.result)) THEN
        RAISE EXCEPTION 'Photo removal command identity and completed result are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER photo_removal_command_identity BEFORE UPDATE ON photo_removal_commands
    FOR EACH ROW EXECUTE FUNCTION protect_photo_removal_command_identity();
