-- One-shot nuke for the two test events.
-- DELETEs use ON DELETE CASCADE for the embed tables (photo_bibs,
-- photo_face_persons). The explicit DELETEs are defense-in-depth in case
-- the cascade isn't present on the schema.
BEGIN;

DELETE FROM photo_bibs WHERE photo_id IN (
  SELECT id FROM photos WHERE event_id IN (
    '7ef1031d-273f-464c-846b-bf267216b259',
    '85bdd891-046a-445f-a4e6-e297200a0e1e'
  )
);

DELETE FROM photo_face_persons WHERE photo_id IN (
  SELECT id FROM photos WHERE event_id IN (
    '7ef1031d-273f-464c-846b-bf267216b259',
    '85bdd891-046a-445f-a4e6-e297200a0e1e'
  )
);

DELETE FROM photos WHERE event_id IN (
  '7ef1031d-273f-464c-846b-bf267216b259',
  '85bdd891-046a-445f-a4e6-e297200a0e1e'
);

UPDATE events SET photo_count = 0 WHERE id IN (
  '7ef1031d-273f-464c-846b-bf267216b259',
  '85bdd891-046a-445f-a4e6-e297200a0e1e'
);

-- Free the event_photographer rows too so the photographer's "events covered"
-- list reflects the clean state.
DELETE FROM event_photographer WHERE event_id IN (
  '7ef1031d-273f-464c-846b-bf267216b259',
  '85bdd891-046a-445f-a4e6-e297200a0e1e'
);

COMMIT;
