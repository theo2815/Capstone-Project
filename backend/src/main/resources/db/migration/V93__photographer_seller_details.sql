-- Seller details for RA 11967 Sec. 21(b) and BIR RR 16-2023: one record per photographer.
-- No ON DELETE: these are legal records, so a users delete fails loudly instead of removing them.
CREATE TABLE photographer_seller_details (
    user_id          UUID         PRIMARY KEY REFERENCES users (id),
    legal_name       VARCHAR(200) NOT NULL,
    id_type          VARCHAR(20)  NOT NULL
                       CHECK (id_type IN ('PHILSYS','DRIVERS_LICENSE','PASSPORT','UMID','PRC','POSTAL','VOTERS')),
    id_front_key     VARCHAR(512) NOT NULL,
    id_back_key      VARCHAR(512),
    selfie_key       VARCHAR(512) NOT NULL,
    bir_form_key     VARCHAR(512),
    address          VARCHAR(500) NOT NULL,
    phone            VARCHAR(30)  NOT NULL,
    status           VARCHAR(20)  NOT NULL CHECK (status IN ('SUBMITTED','APPROVED','REJECTED')),
    rejection_reason VARCHAR(500),
    revision         BIGINT       NOT NULL,
    submitted_at     TIMESTAMPTZ  NOT NULL,
    reviewed_at      TIMESTAMPTZ,
    reviewed_by      UUID         REFERENCES users (id),
    CONSTRAINT chk_seller_details_philsys_back CHECK (id_type <> 'PHILSYS' OR id_back_key IS NOT NULL),
    CONSTRAINT chk_seller_details_rejection_reason CHECK (status <> 'REJECTED' OR rejection_reason IS NOT NULL)
);
CREATE INDEX idx_seller_details_review_queue ON photographer_seller_details (status, submitted_at);
