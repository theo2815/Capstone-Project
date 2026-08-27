-- V34: timestamp of the most recent failed login (NFR-S-14 window).
-- The V29 lockout counted CONSECUTIVE failures with no time bound; the SRS
-- target is "5 failed attempts within 15 min". LoginAttemptService now resets
-- a streak whose last failure is older than the window. Nullable: accounts
-- with no failure history (or a pre-V34 streak) simply start a fresh streak.
ALTER TABLE users
    ADD COLUMN last_failed_login_at TIMESTAMPTZ;
