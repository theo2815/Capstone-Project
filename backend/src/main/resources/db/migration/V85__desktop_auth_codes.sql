-- Desktop connect (BatchMyPhotos upload workflow, spec 2026-10-10): one-time
-- codes the signed-in website mints and the desktop app redeems once, with a
-- PKCE challenge. Only the SHA-256 of the code is stored.
-- ponytail: no purge job — rows are tiny and connects are rare; add a scheduled
-- DELETE of used or expired rows if this table ever grows.
CREATE TABLE desktop_auth_codes (
    id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    code_hash      VARCHAR(255) NOT NULL UNIQUE,
    code_challenge VARCHAR(255) NOT NULL,
    expires_at     TIMESTAMPTZ  NOT NULL,
    used_at        TIMESTAMPTZ,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_desktop_auth_codes_user_id ON desktop_auth_codes (user_id);
