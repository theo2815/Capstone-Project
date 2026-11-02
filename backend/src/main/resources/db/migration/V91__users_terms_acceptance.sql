-- V91: which Privacy Policy / Terms version a user agreed to at sign-up, and when.
-- The server stamps its own app.legal.current-version when a new sign-up sends
-- termsAccepted=true (register, or the Google role step). Nullable: existing
-- users and old app builds never agreed through the app, and nothing requires it.
ALTER TABLE users
    ADD COLUMN terms_version VARCHAR(32);
ALTER TABLE users
    ADD COLUMN terms_accepted_at TIMESTAMPTZ;
