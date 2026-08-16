-- V30: email verification on registration.
--
-- Advisory only. Nothing in the application blocks on email_verified_at — both
-- clients sign a user straight in after /auth/register, so a gate here would
-- strand them on a wall neither front end has copy for. This migration records
-- the fact; enforcing it is a separate, cross-module decision.
--
-- Existing rows stay NULL rather than being backfilled as verified. NULL is the
-- truth: nobody confirmed those addresses. The bootstrap admin is the one
-- exception and it is stamped in application code (BootstrapAdminRunner), since
-- that address is operator-provisioned from env and often isn't a real inbox.
--
-- The token table is email_change_tokens (V28) minus new_email: same opaque
-- token hashed with SHA-256, same single-use + expiry semantics, same two
-- indexes (lookup by hash on redemption, by user so a resend can retire the
-- outstanding one).

ALTER TABLE users
    ADD COLUMN email_verified_at TIMESTAMPTZ;

CREATE TABLE email_verification_tokens (
    id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ  NOT NULL,
    used_at    TIMESTAMPTZ,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_email_verification_tokens_token_hash ON email_verification_tokens (token_hash);
CREATE INDEX idx_email_verification_tokens_user_id    ON email_verification_tokens (user_id);
