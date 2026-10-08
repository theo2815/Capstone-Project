CREATE TABLE admin_action_commands (
    command_id UUID PRIMARY KEY,
    actor_id UUID NOT NULL,
    operation VARCHAR(24) NOT NULL CHECK (operation IN ('USER_MESSAGE', 'EVENT_CREATE', 'EVENT_UPDATE')),
    target_id UUID,
    input_hash VARCHAR(64) NOT NULL CHECK (input_hash ~ '^[0-9a-f]{64}$'),
    state VARCHAR(16) NOT NULL CHECK (state IN ('PENDING', 'UNKNOWN', 'SUCCEEDED', 'FAILED')),
    claim_owner UUID NOT NULL,
    lease_expires_at TIMESTAMPTZ,
    intended_event_id UUID,
    intended_storage_key VARCHAR(512),
    processed_cover_digest VARCHAR(64) CHECK (processed_cover_digest ~ '^[0-9a-f]{64}$'),
    result_resource_id UUID,
    result_decision_id UUID,
    result_message_id UUID,
    claimed_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    CHECK (state <> 'SUCCEEDED' OR (completed_at IS NOT NULL AND result_resource_id IS NOT NULL)),
    CHECK (state <> 'SUCCEEDED' OR operation <> 'USER_MESSAGE' OR
        (result_decision_id IS NOT NULL AND result_message_id IS NOT NULL)),
    CHECK (operation <> 'USER_MESSAGE' OR (target_id IS NOT NULL AND intended_event_id IS NULL
        AND intended_storage_key IS NULL AND processed_cover_digest IS NULL AND lease_expires_at IS NULL))
);

CREATE FUNCTION protect_admin_action_command() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Admin command receipts require an explicit retention policy' USING ERRCODE = '23514';
    END IF;
    IF ROW(NEW.command_id, NEW.actor_id, NEW.operation, NEW.target_id, NEW.input_hash,
           NEW.intended_event_id, NEW.intended_storage_key, NEW.processed_cover_digest, NEW.claimed_at)
        IS DISTINCT FROM ROW(OLD.command_id, OLD.actor_id, OLD.operation, OLD.target_id, OLD.input_hash,
           OLD.intended_event_id, OLD.intended_storage_key, OLD.processed_cover_digest, OLD.claimed_at)
        OR (OLD.state = 'SUCCEEDED' AND NEW IS DISTINCT FROM OLD) THEN
        RAISE EXCEPTION 'Admin command identity and completed results are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER admin_action_command_identity BEFORE UPDATE OR DELETE ON admin_action_commands
    FOR EACH ROW EXECUTE FUNCTION protect_admin_action_command();
