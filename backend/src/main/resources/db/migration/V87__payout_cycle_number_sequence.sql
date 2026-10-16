-- V87: payout cycle ids end in a number no other photographer can share.
-- Request ids used to end in the photographer's own request count, so two
-- handles that clean up to the same letters (john-doe, johndoe), or a handle
-- reused after a rename, built the same id, and save() merged the second
-- request onto the first photographer's row (security audit 2026-10-14).
-- New ids end in this database-wide number. It starts at 100000 so a new id
-- can never equal an old three-digit one. Existing ids are not changed: they
-- are referenced by payout_reports, payout_status_history, admin_decision_log
-- and, for transfers already sent, by PayMongo's idempotency key.
CREATE SEQUENCE payout_cycle_number_seq START WITH 100000;
