-- 2026-10-29: the PayMongo refund link a QR Ph runner claims their refund with,
-- and when it was last sent. Both stay NULL until a link exists; a new link
-- replaces the old one.
ALTER TABLE disputes
    ADD COLUMN refund_link_url     VARCHAR(500),
    ADD COLUMN refund_link_sent_at TIMESTAMPTZ;
