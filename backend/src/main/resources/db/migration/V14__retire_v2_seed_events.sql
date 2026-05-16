-- V14: retire the V2 demo seed events.
--
-- V2 shipped four demo rows so /events / /admin/events had something to
-- render on first boot (cebu-city-marathon-2026, mactan-bridge-run-2026,
-- queen-city-night-run-2026, bohol-trail-classic-2025). With admin event
-- create wired up, those seeds are now noise on top of real admin-created
-- rows — runners would see the demo cards alongside the real catalog.
--
-- Soft-delete (deletedAt) instead of hard DELETE so:
--   - Every EventRepository query already filters `WHERE deleted_at IS
--     NULL`, so the rows vanish from all surfaces with no other code
--     change.
--   - Future-dated CASCADE writes (orders, photos, cart items, saved
--     events, photographer coverage) referencing one of these UUIDs in a
--     local dev DB don't blow up — the seed rows are still there as
--     orphans, not removed.
--   - A fresh DB never sees these seeds because V2 itself isn't being
--     edited (Flyway would refuse to re-validate).

UPDATE events
SET deleted_at = now(),
    updated_at = now()
WHERE id IN (
    '11111111-1111-1111-1111-111111111111'::uuid,
    '22222222-2222-2222-2222-222222222222'::uuid,
    '33333333-3333-3333-3333-333333333333'::uuid,
    '44444444-4444-4444-4444-444444444444'::uuid
)
AND deleted_at IS NULL;
