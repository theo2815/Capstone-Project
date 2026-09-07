-- V57: admin MFA (TOTP) — opt-in, per-admin (Phase 5).
--
-- admin_mfa holds one row per admin who has started enrollment; secret_enc is
-- the AES-256-GCM-encrypted TOTP secret (key derived from JWT_SECRET). enabled
-- flips true only after the admin confirms a code. Recovery codes are stored
-- hashed (SHA-256), single-use via used_at.

CREATE TABLE admin_mfa (
    user_id       UUID          PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    secret_enc    TEXT          NOT NULL,
    enabled       BOOLEAN       NOT NULL DEFAULT false,
    confirmed_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE TABLE admin_mfa_recovery_codes (
    id          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID          NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    code_hash   VARCHAR(100)  NOT NULL,
    used_at     TIMESTAMPTZ,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX idx_admin_mfa_recovery_user ON admin_mfa_recovery_codes (user_id, used_at);
