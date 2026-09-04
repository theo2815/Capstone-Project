-- V29: temporary account lockout after N consecutive failed logins.
--
-- The per-IP Bucket4j policy on /auth/login (10/min) only slows one source, and
-- RATE_LIMIT_ENABLED defaults to false — so a credential-stuffing run spread
-- across hosts was unthrottled per account. These two columns move the counter
-- onto the account itself, where the attacker's IP doesn't matter.
--
-- Deliberately temporary: locked_until auto-clears, so a hostile lockout costs
-- the victim minutes, not an account. NIST SP 800-63B (already this service's
-- reference for PasswordValidator) argues against indefinite lockout for
-- exactly that reason.
--
-- No index. users is only ever reached by primary key or the existing
-- idx_users_email, and both columns are read from an already-loaded row.

ALTER TABLE users
    ADD COLUMN failed_login_attempts INTEGER     NOT NULL DEFAULT 0,
    ADD COLUMN locked_until          TIMESTAMPTZ;
