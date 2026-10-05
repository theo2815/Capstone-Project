-- Case records survive deletion of live events, photos and accounts.
CREATE TABLE photo_removal_requests (
    id UUID PRIMARY KEY,
    event_id UUID REFERENCES events(id) ON DELETE SET NULL,
    event_id_snapshot UUID NOT NULL,
    event_name_snapshot VARCHAR(255) NOT NULL,
    requester_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    requester_contact TEXT NOT NULL CHECK (btrim(requester_contact) <> ''),
    support_reference TEXT NOT NULL CHECK (btrim(support_reference) <> ''),
    requester_reason TEXT NOT NULL CHECK (btrim(requester_reason) <> ''),
    public_to_owner_reason TEXT NOT NULL CHECK (btrim(public_to_owner_reason) <> ''),
    status VARCHAR(16) NOT NULL DEFAULT 'DRAFT'
        CHECK (status IN ('DRAFT', 'OPEN', 'RESOLVING', 'RESOLVED', 'CANCELLED')),
    version BIGINT NOT NULL DEFAULT 0 CHECK (version >= 0),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ,
    creator_id UUID NOT NULL,
    verifier_id UUID,
    verification_method TEXT,
    verification_note TEXT,
    verified_at TIMESTAMPTZ,
    evidence_key VARCHAR(512),
    evidence_content_type VARCHAR(100),
    evidence_generation BIGINT NOT NULL DEFAULT 0 CHECK (evidence_generation >= 0),
    evidence_cleanup_state VARCHAR(16) NOT NULL DEFAULT 'NONE'
        CHECK (evidence_cleanup_state IN ('NONE', 'PRESENT', 'PENDING', 'DELETED', 'RETRY_REQUIRED')),
    outstanding_evidence_keys JSONB NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(outstanding_evidence_keys) = 'array'),
    creation_command_id UUID NOT NULL UNIQUE,
    CHECK (event_id IS NULL OR event_id = event_id_snapshot)
);
CREATE INDEX idx_photo_removal_requests_event_status ON photo_removal_requests(event_id_snapshot, status, created_at, id);

CREATE TABLE photo_removal_reviews (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES photo_removal_requests(id),
    owner_id_snapshot UUID NOT NULL,
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    notified_message_id UUID REFERENCES photographer_messages(id) ON DELETE SET NULL,
    notified_at TIMESTAMPTZ,
    deadline TIMESTAMPTZ,
    overdue BOOLEAN NOT NULL DEFAULT false,
    overdue_at TIMESTAMPTZ,
    unavailable_reason TEXT,
    UNIQUE (request_id, owner_id_snapshot),
    UNIQUE (id, request_id),
    CHECK (owner_id IS NULL OR owner_id = owner_id_snapshot)
);
CREATE INDEX idx_photo_removal_reviews_due ON photo_removal_reviews(deadline, id)
    WHERE notified_at IS NOT NULL AND overdue = false AND unavailable_reason IS NULL;

CREATE TABLE photo_removal_items (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES photo_removal_requests(id),
    photo_id_snapshot UUID NOT NULL,
    photo_id UUID REFERENCES photos(id) ON DELETE SET NULL,
    owner_id_snapshot UUID,
    owner_name_snapshot VARCHAR(255),
    review_id UUID,
    state VARCHAR(24) NOT NULL DEFAULT 'CANDIDATE'
        CHECK (state IN ('CANDIDATE', 'VERIFIED', 'EXCLUDED', 'REMOVAL_AUTHORIZED', 'REMOVED', 'RETAINED', 'RETAINED_RESTRICTED')),
    match_source VARCHAR(10) NOT NULL CHECK (match_source IN ('SELFIE', 'BIB', 'MANUAL')),
    match_score DOUBLE PRECISION,
    verification_actor_id UUID,
    verified_at TIMESTAMPTZ,
    verification_reason TEXT,
    owner_response VARCHAR(20) CHECK (owner_response IN ('AGREE', 'OBJECT', 'ALREADY_DELETED')),
    owner_response_reason TEXT,
    owner_responded_at TIMESTAMPTZ,
    admin_decision VARCHAR(20) CHECK (admin_decision IN ('REMOVE', 'RETAIN', 'MORE_EVIDENCE', 'RETAIN_RESTRICTED')),
    admin_decision_reason TEXT,
    admin_decision_actor_id UUID,
    admin_decided_at TIMESTAMPTZ,
    retention_review_at TIMESTAMPTZ,
    cleanup_job_id UUID,
    UNIQUE (request_id, photo_id_snapshot),
    UNIQUE (id, request_id),
    FOREIGN KEY (review_id, request_id) REFERENCES photo_removal_reviews(id, request_id),
    CHECK (photo_id IS NULL OR photo_id = photo_id_snapshot),
    CHECK (admin_decision IS NULL OR (admin_decision_reason IS NOT NULL AND btrim(admin_decision_reason) <> '')),
    CHECK (state <> 'RETAINED_RESTRICTED' OR retention_review_at IS NOT NULL),
    CHECK (owner_response <> 'OBJECT' OR (owner_response_reason IS NOT NULL AND btrim(owner_response_reason) <> ''))
);
CREATE INDEX idx_photo_removal_items_photo_state ON photo_removal_items(photo_id_snapshot, state);
CREATE INDEX idx_photo_removal_items_review ON photo_removal_items(review_id, id);

CREATE TABLE photo_removal_history (
    id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES photo_removal_requests(id),
    item_id UUID,
    review_id UUID,
    photo_id_snapshot UUID,
    actor_type VARCHAR(16) NOT NULL CHECK (actor_type IN ('ADMIN', 'PHOTOGRAPHER', 'SYSTEM', 'SUPPORT')),
    actor_id_snapshot UUID,
    action VARCHAR(24) NOT NULL CHECK (action IN ('CREATED', 'EVIDENCE_REPLACED', 'VERIFIED', 'NOTICE_STORED',
        'OWNER_UNAVAILABLE', 'OWNER_RESPONDED', 'REVIEW_OVERDUE', 'ADMIN_DECIDED', 'ERASURE_PROGRESS',
        'EVIDENCE_PURGED', 'SUPPORT_UPDATED', 'CANCEL_REQUESTED', 'RESOLVED', 'CANCELLED')),
    from_state VARCHAR(24),
    to_state VARCHAR(24),
    reason TEXT CHECK (reason IS NULL OR btrim(reason) <> ''),
    occurred_at TIMESTAMPTZ NOT NULL,
    command_id UUID NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}' CHECK (jsonb_typeof(metadata) = 'object'),
    FOREIGN KEY (item_id, request_id) REFERENCES photo_removal_items(id, request_id),
    FOREIGN KEY (review_id, request_id) REFERENCES photo_removal_reviews(id, request_id),
    CONSTRAINT uq_photo_removal_history_action UNIQUE NULLS NOT DISTINCT (request_id, command_id, item_id, review_id)
);
CREATE INDEX idx_photo_removal_history_request_time ON photo_removal_history(request_id, occurred_at, id);

CREATE FUNCTION reject_photo_removal_history_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Photo removal history is append-only' USING ERRCODE = '23514';
END;
$$;
CREATE TRIGGER photo_removal_history_append_only BEFORE UPDATE OR DELETE ON photo_removal_history
    FOR EACH ROW EXECUTE FUNCTION reject_photo_removal_history_mutation();

CREATE FUNCTION protect_photo_removal_snapshots() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_TABLE_NAME = 'photo_removal_requests' THEN
        IF ROW(NEW.event_id_snapshot, NEW.event_name_snapshot, NEW.creator_id, NEW.creation_command_id)
            IS DISTINCT FROM ROW(OLD.event_id_snapshot, OLD.event_name_snapshot, OLD.creator_id, OLD.creation_command_id) THEN
            RAISE EXCEPTION 'Photo removal request snapshots are immutable' USING ERRCODE = '23514';
        END IF;
    ELSIF TG_TABLE_NAME = 'photo_removal_items' THEN
        IF ROW(NEW.request_id, NEW.photo_id_snapshot, NEW.owner_id_snapshot, NEW.owner_name_snapshot)
            IS DISTINCT FROM ROW(OLD.request_id, OLD.photo_id_snapshot, OLD.owner_id_snapshot, OLD.owner_name_snapshot) THEN
            RAISE EXCEPTION 'Photo removal item snapshots are immutable' USING ERRCODE = '23514';
        END IF;
    ELSIF NEW.request_id IS DISTINCT FROM OLD.request_id OR NEW.owner_id_snapshot IS DISTINCT FROM OLD.owner_id_snapshot THEN
        RAISE EXCEPTION 'Photo removal review snapshots are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER photo_removal_request_snapshots BEFORE UPDATE ON photo_removal_requests
    FOR EACH ROW EXECUTE FUNCTION protect_photo_removal_snapshots();
CREATE TRIGGER photo_removal_item_snapshots BEFORE UPDATE ON photo_removal_items
    FOR EACH ROW EXECUTE FUNCTION protect_photo_removal_snapshots();
CREATE TRIGGER photo_removal_review_snapshots BEFORE UPDATE ON photo_removal_reviews
    FOR EACH ROW EXECUTE FUNCTION protect_photo_removal_snapshots();
