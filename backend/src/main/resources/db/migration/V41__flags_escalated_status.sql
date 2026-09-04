-- V41: Extend `flags.status` with 'escalated' so flags can be escalated to high-priority triage.
ALTER TABLE flags
    DROP CONSTRAINT IF EXISTS flags_status_check;

ALTER TABLE flags
    ADD CONSTRAINT flags_status_check
        CHECK (status IN ('open', 'resolved', 'hidden', 'dismissed', 'escalated'));
