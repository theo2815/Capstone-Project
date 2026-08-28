-- Google sign-in: identity link for "Continue with Google" (/auth/google).
-- NULL for password-only accounts; set on first Google sign-in (new account
-- or auto-link by verified email). UNIQUE doubles as the sub -> user lookup
-- index; Postgres treats NULLs as distinct, so unlinked rows don't collide.
ALTER TABLE users ADD COLUMN google_sub VARCHAR(255);
ALTER TABLE users ADD CONSTRAINT uq_users_google_sub UNIQUE (google_sub);
