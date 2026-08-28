-- V37: password reset moves from an emailed link token to a 6-digit OTP.
-- One row now serves both phases of a reset: it is born holding the SHA-256
-- of the mailed code (code_hash, deliberately NON-unique — a 6-digit space
-- collides across users), and on successful verification rotates into the
-- continuation token consumed by /auth/reset-password (token_hash, whose
-- UNIQUE constraint stays; it is NULL until verification, so the confirm
-- lookup can never match an unverified code row).
-- attempts caps online guessing per code (dead at 5 regardless of
-- correctness) — a 10^6 space needs a per-code budget, not just the per-IP
-- bucket.
-- user_id index: verify + invalidate-outstanding now query by user on every
-- request; V1 shipped this table with an index on token_hash only.
ALTER TABLE password_reset_tokens ALTER COLUMN token_hash DROP NOT NULL;
ALTER TABLE password_reset_tokens ADD COLUMN code_hash VARCHAR(255);
ALTER TABLE password_reset_tokens ADD COLUMN attempts INT NOT NULL DEFAULT 0;
CREATE INDEX idx_password_reset_tokens_user_id ON password_reset_tokens (user_id);
