-- AA-R1: shared race events (auto-approval Plan 2). A race is an event with event_type set and
-- created_by NULL. Additive and nullable: every existing row stays untyped and unchanged.
-- V95 is unused on purpose (Plan 1 dropped it).

ALTER TABLE events
    ADD COLUMN event_type            VARCHAR(20)
        CONSTRAINT ck_events_event_type
            CHECK (event_type IN ('ROAD_RACE', 'FUN_RUN', 'TRAIL_RUN', 'ULTRA', 'OBSTACLE_RACE', 'CYCLING')),
    ADD COLUMN submitted_by          UUID REFERENCES users (id) ON DELETE SET NULL,
    ADD COLUMN name_key              VARCHAR(200),
    ADD COLUMN possible_duplicate_of UUID REFERENCES events (id) ON DELETE SET NULL,
    ADD COLUMN admin_approved_at     TIMESTAMPTZ,
    ADD COLUMN hidden_at             TIMESTAMPTZ,
    ADD COLUMN proof_checked_at      TIMESTAMPTZ,
    ADD COLUMN check_reasons         JSONB;

-- An untyped row (legacy or "Other") takes no race column; a race is never photographer-owned.
ALTER TABLE events
    ADD CONSTRAINT ck_events_race_only CHECK (
        event_type IS NOT NULL OR (
            submitted_by IS NULL AND name_key IS NULL AND possible_duplicate_of IS NULL
            AND admin_approved_at IS NULL AND hidden_at IS NULL AND proof_checked_at IS NULL
            AND check_reasons IS NULL)),
    ADD CONSTRAINT ck_events_race_shared CHECK (event_type IS NULL OR created_by IS NULL);

-- Other names photographers typed for the same race (filled on join, AA-R3).
CREATE TABLE event_name_aliases (
    event_id   UUID         NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    name_key   VARCHAR(200) NOT NULL,
    added_by   UUID         REFERENCES users (id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, name_key)
);
