ALTER TABLE photographer_settings
    ADD COLUMN verification_revision BIGINT NOT NULL DEFAULT 0,
    ADD CONSTRAINT ck_photographer_verification_revision_nonnegative
        CHECK (verification_revision >= 0);
