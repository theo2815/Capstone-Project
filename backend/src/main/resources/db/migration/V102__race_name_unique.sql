-- AA-R10 (auto-approval Plan 2): one live self-serve race per date and name key, the last line
-- behind RaceEventService's advisory lock (a violation answers 409 RACE_DUPLICATE). Admin creates
-- (submitted_by NULL), "different race" decisions (possible_duplicate_of set), deleted and rejected
-- races stay outside it. Planned as V98; V99 and V100 merged first, and Flyway refuses a lower version.
CREATE UNIQUE INDEX uq_events_race_date_name ON events (date, name_key)
    WHERE event_type IS NOT NULL AND submitted_by IS NOT NULL AND possible_duplicate_of IS NULL
      AND deleted_at IS NULL AND review_status <> 'REJECTED';
