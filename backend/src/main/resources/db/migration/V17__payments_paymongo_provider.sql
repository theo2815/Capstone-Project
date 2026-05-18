-- V17 — Allow `paymongo` as a payments.provider value and rebuild the
-- (provider, provider_ref) uniqueness to include order_id.
--
-- Two changes:
--
--   1. Extend the payments.provider CHECK constraint to accept 'paymongo'.
--      Real PayMongo Payment rows carry provider='paymongo' (the gateway);
--      the user's wallet preference stays on orders.payment_method
--      (gcash / maya / card). The old values keep working for the legacy
--      stub path and any future direct-integration providers.
--
--   2. Rebuild the (provider, provider_ref) uniqueness as
--      (provider, provider_ref, order_id). A single PayMongo Checkout
--      Session covers a multi-event cart — OrderService.create splits by
--      event into N orders, and all N Payment rows share the same cs_id
--      as providerRef. The old UNIQUE collapsed those into one row;
--      including order_id keeps each order's payment row distinct while
--      preserving the per-order webhook-replay guard.

ALTER TABLE payments
    DROP CONSTRAINT payments_provider_check;

ALTER TABLE payments
    ADD CONSTRAINT payments_provider_check
        CHECK (provider IN ('gcash', 'maya', 'card', 'stub', 'paymongo'));

DROP INDEX IF EXISTS uq_payments_provider_ref;

CREATE UNIQUE INDEX uq_payments_provider_ref
    ON payments (provider, provider_ref, order_id)
    WHERE provider_ref IS NOT NULL;
