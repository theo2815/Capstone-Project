-- Change-email flow (PUT /me/email). Mirrors password_reset_tokens: an opaque
-- token, stored only as a SHA-256 hash, single-use, short-lived.
--
-- The extra column is new_email. The address is parked HERE, not on users, so
-- an unconfirmed request never affects sign-in: users.email only changes when
-- the token is redeemed from the new inbox. That is what stops a stolen
-- session from silently reassigning the account.

CREATE TABLE email_change_tokens (
    id         UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    new_email  VARCHAR(255) NOT NULL,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ  NOT NULL,
    used_at    TIMESTAMPTZ,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_email_change_tokens_token_hash ON email_change_tokens (token_hash);

-- Lets a re-request cheaply invalidate a user's outstanding tokens so only the
-- newest confirmation link works.
CREATE INDEX idx_email_change_tokens_user_id ON email_change_tokens (user_id);
