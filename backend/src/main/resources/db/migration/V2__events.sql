-- V2: events catalog
-- One row per organized event. Photos / participants / event_photographer
-- arrive in V3 (PR 3) and V7 (PR 7).

CREATE TABLE events (
    id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    slug              VARCHAR(160) UNIQUE NOT NULL,
    name              VARCHAR(255) NOT NULL,
    date              DATE         NOT NULL,
    location          VARCHAR(255) NOT NULL,
    banner_url        VARCHAR(512),
    photo_count       INTEGER      NOT NULL DEFAULT 0,
    participant_count INTEGER      NOT NULL DEFAULT 0,
    status            VARCHAR(20)  NOT NULL CHECK (status IN ('DRAFT', 'ACTIVE', 'COMPLETED', 'ARCHIVED')),
    description       TEXT         NOT NULL DEFAULT '',
    organizer_name    VARCHAR(255) NOT NULL DEFAULT '',
    price_per_photo   NUMERIC(12, 2) NOT NULL DEFAULT 0,
    bundle_price      NUMERIC(12, 2),
    bundle_size       INTEGER,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    deleted_at        TIMESTAMPTZ
);

CREATE INDEX idx_events_date_id ON events (date DESC, id);
CREATE INDEX idx_events_status  ON events (status);

CREATE TABLE event_categories (
    event_id UUID         NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    category VARCHAR(50)  NOT NULL,
    PRIMARY KEY (event_id, category)
);

CREATE INDEX idx_event_categories_category ON event_categories (category);

-- Seed: a few events so the runner /events page renders something on first
-- BACKEND_LIVE=true smoke. Mock catalog (lib/event-catalog.ts) lives only
-- on the FE side until PR 7 photographer uploads start populating real data.

INSERT INTO events (id, slug, name, date, location, banner_url, photo_count, participant_count, status, description, organizer_name, price_per_photo)
VALUES
  ('11111111-1111-1111-1111-111111111111'::uuid, 'cebu-city-marathon-2026', 'Cebu City Marathon 2026', DATE '2026-05-17', 'Cebu Business Park, Cebu City', NULL, 0, 0, 'ACTIVE', 'The flagship 42K through Cebu''s old town and reclamation routes.', 'CCM Organizing Committee', 125),
  ('22222222-2222-2222-2222-222222222222'::uuid, 'mactan-bridge-run-2026', 'Mactan Bridge Run 2026', DATE '2026-06-07', 'Mactan-Cordova Bridge, Lapu-Lapu City', NULL, 0, 0, 'ACTIVE', 'Sunrise 21K crossing the Mactan-Cordova Expressway.', 'CCLEX Sports Club', 125),
  ('33333333-3333-3333-3333-333333333333'::uuid, 'queen-city-night-run-2026', 'Queen City Night Run 2026', DATE '2026-04-28', 'IT Park, Cebu City', NULL, 0, 0, 'COMPLETED', 'After-dark 10K and 5K through the IT Park loop.', 'Queen City Runners', 125),
  ('44444444-4444-4444-4444-444444444444'::uuid, 'bohol-trail-classic-2025', 'Bohol Trail Classic 2025', DATE '2025-11-08', 'Loboc River Resort, Bohol', NULL, 0, 0, 'ARCHIVED', 'Trail-focused 25K and 12K winding through the Loboc river system.', 'Bohol Trail Society', 125);

INSERT INTO event_categories (event_id, category) VALUES
  ('11111111-1111-1111-1111-111111111111'::uuid, '5K'),
  ('11111111-1111-1111-1111-111111111111'::uuid, '10K'),
  ('11111111-1111-1111-1111-111111111111'::uuid, '21K'),
  ('11111111-1111-1111-1111-111111111111'::uuid, '42K'),
  ('22222222-2222-2222-2222-222222222222'::uuid, '5K'),
  ('22222222-2222-2222-2222-222222222222'::uuid, '10K'),
  ('22222222-2222-2222-2222-222222222222'::uuid, '21K'),
  ('33333333-3333-3333-3333-333333333333'::uuid, '5K'),
  ('33333333-3333-3333-3333-333333333333'::uuid, '10K'),
  ('44444444-4444-4444-4444-444444444444'::uuid, '12K'),
  ('44444444-4444-4444-4444-444444444444'::uuid, '25K');
