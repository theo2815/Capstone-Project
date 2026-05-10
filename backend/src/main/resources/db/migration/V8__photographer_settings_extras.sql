-- V8: photographer settings extras (PR 8 / Phase F.2)
-- social_links: per-photographer outbound URLs (≥1 required for verification).
-- photographer_payout_accounts: payout destinations (≥1 required, exactly one
-- primary). Partial unique index enforces the single-primary invariant without
-- a trigger — Postgres won't let two rows for the same user_id be is_primary=true.

-- Enum-as-string columns store the JPA EnumType.STRING value (uppercase enum
-- name). Service code maps to/from the lowercase wire form via SocialPlatform.wire
-- and PayoutMethod.wire, matching the verification_status convention from V7.
CREATE TABLE social_links (
    id          UUID         PRIMARY KEY,
    user_id     UUID         NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    platform    VARCHAR(20)  NOT NULL
                  CHECK (platform IN ('FACEBOOK','INSTAGRAM','TIKTOK','X','YOUTUBE','WEBSITE')),
    url         VARCHAR(500) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_social_links_user ON social_links (user_id, created_at);

CREATE TABLE photographer_payout_accounts (
    id              UUID          PRIMARY KEY,
    user_id         UUID          NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    method          VARCHAR(20)   NOT NULL
                      CHECK (method IN ('GCASH','MAYA','GOTYME')),
    account_number  VARCHAR(40)   NOT NULL,
    account_name    VARCHAR(100)  NOT NULL,
    qr_s3_key       VARCHAR(512),
    is_primary      BOOLEAN       NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX idx_payout_accounts_user ON photographer_payout_accounts (user_id, created_at);

-- Single-primary invariant: only one row per user can have is_primary=true.
CREATE UNIQUE INDEX uq_payout_accounts_primary_per_user
    ON photographer_payout_accounts (user_id)
    WHERE is_primary = true;
