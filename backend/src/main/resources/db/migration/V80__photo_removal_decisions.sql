ALTER TABLE photo_removal_commands DROP CONSTRAINT photo_removal_commands_operation_check;
ALTER TABLE photo_removal_commands ADD CONSTRAINT photo_removal_commands_operation_check
    CHECK (operation IN ('CREATE', 'REPLACE_EVIDENCE', 'CANCEL', 'MATCH_CANDIDATES', 'VERIFY', 'RESPOND',
        'DECIDE', 'RETRY_CLEANUP', 'SUPPORT_UPDATE'));

-- Completed commands keep their identity; only terminal privacy minimization may change their result.
CREATE FUNCTION sanitize_photo_removal_result(operation TEXT, result JSONB) RETURNS JSONB
LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE item JSONB; cleaned JSONB := '[]'; field TEXT; path TEXT[];
BEGIN
    IF operation = 'MATCH_CANDIDATES' THEN
        RETURN jsonb_set(result, '{items}', '[]') || '{"candidatesPurged":true}';
    END IF;
    IF jsonb_typeof(result->'items') IS DISTINCT FROM 'array' THEN RETURN result; END IF;
    FOR item IN SELECT value FROM jsonb_array_elements(result->'items') LOOP
        FOREACH field IN ARRAY ARRAY['imageUrl', 'cleanUrl', 'downloadUrl'] LOOP
            path := CASE WHEN jsonb_typeof(item#>'{photo,photo}') = 'object'
                THEN ARRAY['photo','photo',field] ELSE ARRAY['photo',field] END;
            IF item#>path IS NOT NULL THEN item := jsonb_set(item, path, 'null'); END IF;
        END LOOP;
        cleaned := cleaned || jsonb_build_array(item);
    END LOOP;
    RETURN jsonb_set(result, '{items}', cleaned);
END;
$$;

CREATE OR REPLACE FUNCTION protect_photo_removal_command_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF ROW(NEW.command_id, NEW.request_id, NEW.operation, NEW.actor_id_snapshot, NEW.input_hash,
           NEW.intended_evidence_key, NEW.claimed_at)
        IS DISTINCT FROM ROW(OLD.command_id, OLD.request_id, OLD.operation, OLD.actor_id_snapshot,
           OLD.input_hash, OLD.intended_evidence_key, OLD.claimed_at)
        OR (OLD.state = 'SUCCEEDED' AND ROW(NEW.state, NEW.result) IS DISTINCT FROM ROW(OLD.state, OLD.result)
            AND NOT (NEW.state = OLD.state
                AND NEW.result = sanitize_photo_removal_result(OLD.operation, OLD.result)
                AND EXISTS (SELECT 1 FROM photo_removal_requests r WHERE r.id = OLD.request_id AND r.status IN ('RESOLVED','CANCELLED')))) THEN
        RAISE EXCEPTION 'Photo removal command identity and completed result are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

-- Aggregate versions order re-review and decisions even when their Clock timestamps are equal.
CREATE FUNCTION photo_removal_owner_review_pending(item_id UUID) RETURNS BOOLEAN
LANGUAGE SQL STABLE AS $$
    SELECT EXISTS (
        SELECT 1 FROM photo_removal_items i WHERE i.id = item_id AND i.state = 'RETAINED_RESTRICTED'
        AND EXISTS (SELECT 1 FROM photo_removal_history v WHERE v.request_id = i.request_id AND v.item_id = i.id
            AND v.review_id IS NOT DISTINCT FROM i.review_id
            AND v.action = 'VERIFIED' AND v.metadata->>'kind' = 'owner_reverification'
            AND CAST(v.metadata->>'noticeVersion' AS bigint) > COALESCE(
                (SELECT max(CAST(d.metadata->>'decisionVersion' AS bigint)) FROM photo_removal_history d
                    WHERE d.request_id = i.request_id AND d.item_id = i.id AND d.action = 'ADMIN_DECIDED'), -1))
    );
$$;
