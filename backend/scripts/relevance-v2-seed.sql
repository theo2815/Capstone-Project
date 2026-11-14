-- relevance-v2-seed.sql: made-up Photo Relevance version 2 results for six staging batches (RV-1.12, G15).
--
-- Who runs it: the owner, by hand, on STAGING only, for the record-only walk-through. Never on production:
-- every row it writes is made up. Staging has AI off, so no real check ever starts there; this script stands in
-- for the AWS and Claude steps, and the backend's own tick then applies the cases for real.
--
-- Input: six batches that already exist, one per verdict. A batch is one photographer's photos in one public
-- event, passed as <event_id>/<photographer_id>:
--   -v fits=<e>/<p> -v mixed=<e>/<p> -v junk_dump=<e>/<p> -v wrong_event=<e>/<p> -v copied=<e>/<p> -v cant_tell=<e>/<p>
--
-- It refuses (error, no change) unless every batch: is named once; has an event that exists, is not deleted,
-- not hidden and Public; has a photographer who exists and is not suspended; has at least one live photo; and
-- has no relevance case, no score row and no word row yet. So every version 2 row of a seeded batch is the
-- seed's own, and relevance-v2-seed-cleanup.sql can remove exactly those.
--
-- What it writes, in one transaction, for each batch:
--   batch_relevance: the batch row, only if version 1 has not made it yet (it is kept by the clean-up).
--   relevance_photo_checks: one made-up measurement per live photo. A "junk" photo is blurry (foreground
--     sharpness 4) and shows no person or race label; the others are sharp race photos. Junk share: MIXED 40%,
--     JUNK_DUMP 90%, every other verdict 0%.
--   batch_relevance_words: two words on every photo (WRONG_EVENT: MANILA, MARATHON; others: CEBU, RUN).
--   relevance_cases: one case, trigger SEEDED, stage DECIDED, the verdict, an admin_reason starting "SEEDED:",
--     with every photo as its sample (relevance_case_photos).
-- It touches no photo, flag, hold, payout or message. It prints what it inserted.
--
-- Run it on the local Docker database (PowerShell, from the repo root):
--   Get-Content backend/scripts/relevance-v2-seed.sql | docker exec -i quickpitik-postgres psql -X -U quickpitik -d quickpitik -v fits=<e>/<p> ...
-- Run it on staging: railway connect (the staging Postgres), then \set fits '<e>/<p>' and the other five, then
-- \i '<repo path with forward slashes>/backend/scripts/relevance-v2-seed.sql'. If it stops with an error inside
-- psql, type ROLLBACK before anything else.

\set ON_ERROR_STOP on

\if :{?fits}
  \set qp_fits :fits
\else
  \set qp_fits ''
\endif
\if :{?mixed}
  \set qp_mixed :mixed
\else
  \set qp_mixed ''
\endif
\if :{?junk_dump}
  \set qp_junk_dump :junk_dump
\else
  \set qp_junk_dump ''
\endif
\if :{?wrong_event}
  \set qp_wrong_event :wrong_event
\else
  \set qp_wrong_event ''
\endif
\if :{?copied}
  \set qp_copied :copied
\else
  \set qp_copied ''
\endif
\if :{?cant_tell}
  \set qp_cant_tell :cant_tell
\else
  \set qp_cant_tell ''
\endif

BEGIN;
SELECT set_config('qp.seed_fits', :'qp_fits', true) AS fits,
       set_config('qp.seed_mixed', :'qp_mixed', true) AS mixed,
       set_config('qp.seed_junk_dump', :'qp_junk_dump', true) AS junk_dump,
       set_config('qp.seed_wrong_event', :'qp_wrong_event', true) AS wrong_event,
       set_config('qp.seed_copied', :'qp_copied', true) AS copied,
       set_config('qp.seed_cant_tell', :'qp_cant_tell', true) AS cant_tell;

CREATE TEMP TABLE seed_targets (
    verdict TEXT, event_id UUID, photographer_id UUID, batch_id UUID, photos INT, junk_percent INT
) ON COMMIT DROP;

DO $$
DECLARE
    v text;
    raw text;
    ev uuid;
    ph uuid;
    bid uuid;
    n int;
BEGIN
    FOREACH v IN ARRAY ARRAY['FITS', 'MIXED', 'JUNK_DUMP', 'WRONG_EVENT', 'COPIED', 'CANT_TELL'] LOOP
        raw := current_setting('qp.seed_' || lower(v));
        IF raw = '' OR split_part(raw, '/', 2) = '' THEN
            RAISE EXCEPTION 'Refused: pass -v %=<event_id>/<photographer_id>', lower(v);
        END IF;
        ev := split_part(raw, '/', 1)::uuid;
        ph := split_part(raw, '/', 2)::uuid;
        IF EXISTS (SELECT 1 FROM seed_targets WHERE event_id = ev AND photographer_id = ph) THEN
            RAISE EXCEPTION 'Refused: batch %/% is named for two verdicts', ev, ph;
        END IF;
        -- The four things the apply step checks again (BatchRelevanceRepository.stillAllowed).
        IF NOT EXISTS (SELECT 1 FROM events WHERE id = ev AND deleted_at IS NULL AND hidden_at IS NULL AND visibility = 'PUBLIC') THEN
            RAISE EXCEPTION 'Refused: % event % is missing, deleted, hidden or not Public', v, ev;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM users WHERE id = ph AND role = 'PHOTOGRAPHER' AND suspended_at IS NULL) THEN
            RAISE EXCEPTION 'Refused: % photographer % is missing, not a photographer, or suspended', v, ph;
        END IF;
        SELECT count(*) INTO n FROM photos
        WHERE event_id = ev AND photographer_id = ph AND deleted_at IS NULL AND removal_state = 'NONE';
        IF n = 0 THEN
            RAISE EXCEPTION 'Refused: % batch %/% has no live photos', v, ev, ph;
        END IF;
        IF EXISTS (SELECT 1 FROM relevance_cases c JOIN batch_relevance br ON br.id = c.batch_id
                   WHERE br.event_id = ev AND br.photographer_id = ph) THEN
            RAISE EXCEPTION 'Refused: % batch %/% already has a relevance case', v, ev, ph;
        END IF;
        IF EXISTS (SELECT 1 FROM relevance_photo_checks r
                   WHERE r.batch_id IN (SELECT id FROM batch_relevance WHERE event_id = ev AND photographer_id = ph)
                      OR r.photo_id IN (SELECT id FROM photos WHERE event_id = ev AND photographer_id = ph)) THEN
            RAISE EXCEPTION 'Refused: % batch %/% already has relevance score rows', v, ev, ph;
        END IF;
        IF EXISTS (SELECT 1 FROM batch_relevance_words w JOIN batch_relevance br ON br.id = w.batch_id
                   WHERE br.event_id = ev AND br.photographer_id = ph) THEN
            RAISE EXCEPTION 'Refused: % batch %/% already has relevance word rows', v, ev, ph;
        END IF;
        INSERT INTO batch_relevance (event_id, photographer_id) VALUES (ev, ph)
        ON CONFLICT (event_id, photographer_id) DO NOTHING;
        SELECT id INTO bid FROM batch_relevance WHERE event_id = ev AND photographer_id = ph;
        INSERT INTO seed_targets VALUES (v, ev, ph, bid, n,
            CASE v WHEN 'MIXED' THEN 40 WHEN 'JUNK_DUMP' THEN 90 ELSE 0 END);
    END LOOP;
END
$$;

-- Made-up measurements: the first junk_percent of each batch's photos are junk.
INSERT INTO relevance_photo_checks (photo_id, batch_id, words_counted_at, measured_at,
                                    fg_sharpness, fg_brightness, sharpness, brightness, person_conf, race_conf)
SELECT p.id, t.batch_id, now(), now(),
       CASE WHEN p.junk THEN 4 ELSE 82 END, 60, CASE WHEN p.junk THEN 6 ELSE 78 END, 60,
       CASE WHEN p.junk THEN 0.05 ELSE 0.98 END, CASE WHEN p.junk THEN 0.03 ELSE 0.91 END
FROM seed_targets t
CROSS JOIN LATERAL (
    SELECT x.id, row_number() OVER (ORDER BY x.id) * 100 <= t.photos * t.junk_percent AS junk
    FROM photos x
    WHERE x.event_id = t.event_id AND x.photographer_id = t.photographer_id
      AND x.deleted_at IS NULL AND x.removal_state = 'NONE'
) p;

INSERT INTO batch_relevance_words (batch_id, word, photos)
SELECT t.batch_id, w.word, t.photos
FROM seed_targets t
CROSS JOIN LATERAL unnest(CASE WHEN t.verdict = 'WRONG_EVENT' THEN ARRAY['MANILA', 'MARATHON']
                               ELSE ARRAY['CEBU', 'RUN'] END) AS w(word);

INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, sample_round, sample_size, measured,
                             blurry, dark, not_race, junk, words_photos, band, copy_low, date_low, decide_attempts,
                             model_verdict, verdict, admin_reason, evidence_used, summary_sent, model, prompt_version, decided_at)
SELECT t.batch_id, 'SEEDED', coalesce(ep.last_upload_at, now()), t.photos, 'DECIDED', 1, t.photos, t.photos,
       j.junk, 0, j.junk, j.junk, t.photos,
       CASE t.verdict WHEN 'MIXED' THEN 'MIXED' WHEN 'JUNK_DUMP' THEN 'OVER' WHEN 'CANT_TELL' THEN 'UNKNOWN' ELSE 'UNDER' END,
       t.verdict = 'COPIED', t.verdict = 'WRONG_EVENT', 1,
       t.verdict, t.verdict,
       'SEEDED: made-up ' || lower(t.verdict) || ' case for the staging walk-through, not a real check.',
       CASE t.verdict WHEN 'COPIED' THEN 'copy_rule' WHEN 'WRONG_EVENT' THEN 'date_rule,batch_words'
                      WHEN 'CANT_TELL' THEN 'junk_share' ELSE 'junk_share' END,
       '{"seeded":true}', 'seeded', 'seeded', now()
FROM seed_targets t
LEFT JOIN event_photographer ep ON ep.event_id = t.event_id AND ep.photographer_id = t.photographer_id
CROSS JOIN LATERAL (SELECT count(*)::int AS junk FROM relevance_photo_checks r
                    WHERE r.batch_id = t.batch_id AND r.fg_sharpness < 10) j;

INSERT INTO relevance_case_photos (case_id, photo_id, round)
SELECT c.id, r.photo_id, 1
FROM seed_targets t
JOIN relevance_cases c ON c.batch_id = t.batch_id AND c.trigger = 'SEEDED'
JOIN relevance_photo_checks r ON r.batch_id = t.batch_id;

\echo '== Inserted: one SEEDED case per batch (stage DECIDED; the next ticks apply them, three a tick)'
SELECT t.verdict, t.event_id, t.photographer_id, t.batch_id, c.id AS case_id, t.photos AS photo_scores,
       c.junk AS junk_photos,
       (SELECT count(*) FROM batch_relevance_words w WHERE w.batch_id = t.batch_id) AS words,
       (SELECT count(*) FROM relevance_case_photos s WHERE s.case_id = c.id) AS sample_photos
FROM seed_targets t
JOIN relevance_cases c ON c.batch_id = t.batch_id AND c.trigger = 'SEEDED'
ORDER BY array_position(ARRAY['FITS', 'MIXED', 'JUNK_DUMP', 'WRONG_EVENT', 'COPIED', 'CANT_TELL'], t.verdict);

COMMIT;
\echo 'SEEDED. Remove every seeded row later with relevance-v2-seed-cleanup.sql.'
