-- RV-L7 part a (legal 1.7): the day an admin recorded that a photographer withdrew the consent to automatic
-- Photo Relevance actions. NULL means not withdrawn. Nullable with no default: no table rewrite, no backfill.
ALTER TABLE users
    ADD COLUMN relevance_consent_withdrawn_at TIMESTAMPTZ;
