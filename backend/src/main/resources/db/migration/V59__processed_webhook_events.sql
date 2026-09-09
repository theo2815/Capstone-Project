-- V59: webhook event-id dedupe (Phase 11)
--
-- Insert-first dedupe for inbound provider webhooks: claim (provider, event_id)
-- before processing, so a redelivered/duplicated event is a no-op. Used by the
-- PayMongo transfer webhook now; available to any future webhook consumer.
-- (The existing payment webhook keeps its own dedupe via the payments UNIQUE +
-- order-status guard — not migrated here to avoid touching that live path.)

CREATE TABLE processed_webhook_events (
    id          UUID PRIMARY KEY,
    provider    VARCHAR(40)  NOT NULL,
    event_id    VARCHAR(120) NOT NULL,
    received_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_processed_webhook_events UNIQUE (provider, event_id)
);
