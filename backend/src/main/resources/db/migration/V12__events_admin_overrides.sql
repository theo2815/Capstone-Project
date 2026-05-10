-- V12: per-row admin override history on events (C-5).
--
-- Plan: .claude-qp/plans/alright-claude-i-recently-ancient-milner.md (§2 C-5).
--
-- Problem: Phase G PATCH /api/v1/admin/events/{id} (Q-A3 — title/date/location
--   only) currently only writes the change shape into admin_decision_log
--   (decision='event_updated', meta={field:{from,to}}). The plan locks
--   `events.admin_overrides JSONB` as a parallel per-event audit trail so the
--   admin /events row can surface override history without joining the
--   decision-log table per-render. The column was missing from V2, so
--   AdminEventService.update had nowhere to mirror the entry.
--
-- Fix: add the column NOT NULL DEFAULT '[]'::jsonb so:
--   - existing rows backfill to an empty array (no NULL handling downstream)
--   - new rows from POST /api/v1/admin/events get '[]' on insert
--   - AdminEventService.update appends {at, adminId, before, after} entries
--     per Q-A3 PATCH, replacing the list reference so Hibernate's
--     dirty-checking notices the change
--
-- Backfill: not required as data, but the DEFAULT clause materialises '[]'
--   for every pre-V12 events row so the entity's non-nullable
--   List<Map<String, Any?>> field never reads NULL.
--
-- The admin_decision_log row written in the same TX (PR 10) stays — that's
-- the cross-event audit trail (filterable by admin_id / decided_at), and
-- events.admin_overrides is its per-event mirror. Both writes are part of
-- the same @Transactional method, so they commit atomically.

ALTER TABLE events
    ADD COLUMN admin_overrides JSONB NOT NULL DEFAULT '[]'::jsonb;
