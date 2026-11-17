-- RVF-3: the Photo Relevance review fixes' one migration. Additive only; the code starts using it in later tasks.

-- RVF-8: the mode a check started under. NULL for every case made before this migration.
ALTER TABLE relevance_cases
    ADD COLUMN started_mode VARCHAR(12) CHECK (started_mode IN ('RECORD_ONLY', 'AUTO_HIDE')),
-- RVF-9: the photographer had not consented when the case was applied (was only the note NO_CONSENT).
    ADD COLUMN no_consent BOOLEAN NOT NULL DEFAULT FALSE;
UPDATE relevance_cases SET no_consent = TRUE WHERE verdict_note = 'NO_CONSENT';

-- RVF-12: an admin restore voids a confirmed case's strike instead of deleting it.
ALTER TABLE relevance_strikes
    ADD COLUMN voided_at TIMESTAMPTZ;

-- RVF-4: a check that ended with no usable result is tried again.
ALTER TABLE relevance_cases DROP CONSTRAINT relevance_cases_trigger_check;
ALTER TABLE relevance_cases ADD CONSTRAINT relevance_cases_trigger_check
    CHECK (trigger IN ('QUIET', 'COUNT', 'RULES', 'SEEDED', 'RETRY'));

-- RVF-7: the model called a batch a wrong event on printed words only.
ALTER TABLE relevance_alerts DROP CONSTRAINT relevance_alerts_kind_check;
ALTER TABLE relevance_alerts ADD CONSTRAINT relevance_alerts_kind_check
    CHECK (kind IN ('BIG_HIDE', 'COPIED', 'APPEAL', 'THIRD_CASE', 'WORDS_ONLY'));
