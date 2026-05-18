-- V16 — Wire orders for PayMongo + Resend email receipts.
--
-- Two new columns on `orders`:
--
--   share_token VARCHAR(64) UNIQUE NULL
--     Opaque token minted at order-create time for GUEST orders so the
--     paid-photos page is reachable from the email receipt without an
--     account (URL shape: /orders/{id}?token=<share_token>). Authed
--     orders leave this NULL — they reach the page via JWT. UNIQUE so
--     a leaked token authenticates exactly one order.
--
--   email_sent_at TIMESTAMPTZ NULL
--     Set when the Resend receipt fires successfully. The webhook
--     handler reads this before sending so PayMongo's retries
--     (checkout_session.payment.paid is delivered until 2xx ack) don't
--     produce duplicate emails. NULL = receipt pending.
--
-- The UNIQUE constraint on share_token creates the lookup index implicitly
-- (PostgreSQL's UNIQUE compiles to a btree). No separate index needed.

ALTER TABLE orders
    ADD COLUMN share_token   VARCHAR(64),
    ADD COLUMN email_sent_at TIMESTAMPTZ;

ALTER TABLE orders
    ADD CONSTRAINT uq_orders_share_token UNIQUE (share_token);
