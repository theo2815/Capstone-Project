-- V4: runner state on top of events + photos.
-- saved_events  : favourites / bookmarks (Q-003 merge on login).
-- cart_items    : pre-purchase basket; snapshot price at add time so server
--                 wins on price changes between add and checkout (Q-003).

CREATE TABLE saved_events (
    user_id  UUID         NOT NULL REFERENCES users  (id) ON DELETE CASCADE,
    event_id UUID         NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    saved_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, event_id)
);

CREATE INDEX idx_saved_events_user ON saved_events (user_id, saved_at DESC);

CREATE TABLE cart_items (
    user_id            UUID         NOT NULL REFERENCES users  (id) ON DELETE CASCADE,
    photo_id           UUID         NOT NULL REFERENCES photos (id) ON DELETE CASCADE,
    event_id           UUID         NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    price_php_at_add   NUMERIC(12, 2) NOT NULL,
    added_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, photo_id)
);

CREATE INDEX idx_cart_items_user ON cart_items (user_id, added_at DESC);
