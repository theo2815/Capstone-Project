-- V88: credential version (security audit 2026-10-14). Goes up by one whenever a
-- user's credentials are reset or changed: the Google takeover guard, an OTP
-- password reset, a password change, an email change. Access tokens carry it as
-- claim `cv`; starting an email change needs a session with the current value,
-- and an email-change link works only while the value it was issued under is
-- current. Existing rows start at 0, which is how tokens without the claim read.
ALTER TABLE users ADD COLUMN credential_version INT NOT NULL DEFAULT 0;
ALTER TABLE email_change_tokens ADD COLUMN credential_version INT NOT NULL DEFAULT 0;
