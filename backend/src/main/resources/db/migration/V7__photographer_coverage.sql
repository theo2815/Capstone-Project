-- V7: photographer coverage + settings + reserved-handle seed
-- photographer_settings: 1:1 with users; lazy-created on first /me/photographer read
-- event_photographer: m:n linking photographers to events with rolling counters
-- reserved_handles: lookup table seeded with system slugs that handles can't shadow

CREATE TABLE photographer_settings (
    user_id              UUID         PRIMARY KEY REFERENCES users (id) ON DELETE CASCADE,
    handle               VARCHAR(40)  UNIQUE,
    brand_name           VARCHAR(100),
    brand_color          VARCHAR(20)  NOT NULL DEFAULT 'none'
                            CHECK (brand_color IN ('none','fresh','amber','indigo','rose','ink')),
    bio                  TEXT         NOT NULL DEFAULT '',
    region_code          VARCHAR(20),
    province_code        VARCHAR(40),
    city                 VARCHAR(100),
    cover_s3_key         VARCHAR(512),
    cover_gradient_from  VARCHAR(20),
    cover_gradient_to    VARCHAR(20),
    watermark_s3_key     VARCHAR(512),
    watermark_label      VARCHAR(40),
    verification_status  VARCHAR(20)  NOT NULL DEFAULT 'INCOMPLETE'
                            CHECK (verification_status IN ('INCOMPLETE','PENDING','APPROVED','REJECTED')),
    member_since         DATE         NOT NULL DEFAULT CURRENT_DATE,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX idx_photographer_settings_handle ON photographer_settings (handle);
CREATE INDEX idx_photographer_settings_verification ON photographer_settings (verification_status);

CREATE TABLE event_photographer (
    event_id           UUID         NOT NULL REFERENCES events (id) ON DELETE CASCADE,
    photographer_id    UUID         NOT NULL REFERENCES users  (id) ON DELETE CASCADE,
    joined_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    photo_count        INTEGER      NOT NULL DEFAULT 0,
    sales_count        INTEGER      NOT NULL DEFAULT 0,
    revenue_kept_php   NUMERIC(14,2) NOT NULL DEFAULT 0,
    first_upload_at    TIMESTAMPTZ,
    last_upload_at     TIMESTAMPTZ,
    PRIMARY KEY (event_id, photographer_id)
);

CREATE INDEX idx_event_photographer_photographer ON event_photographer (photographer_id, last_upload_at DESC NULLS LAST);

CREATE TABLE reserved_handles (
    handle VARCHAR(40) PRIMARY KEY
);

INSERT INTO reserved_handles (handle) VALUES
    ('admin'), ('administrator'), ('support'), ('help'), ('api'),
    ('www'), ('mail'), ('email'), ('quickpitik'), ('staff'),
    ('moderator'), ('mod'), ('billing'), ('payments'), ('legal'),
    ('terms'), ('privacy'), ('about'), ('contact'), ('jobs'),
    ('careers'), ('press'), ('blog'), ('home'), ('login'),
    ('register'), ('signup'), ('signin'), ('logout'),
    ('settings'), ('account'), ('profile'), ('dashboard'),
    ('events'), ('photos'), ('p'), ('public'), ('private'),
    ('me'), ('user'), ('users'), ('null'), ('undefined'),
    ('root'), ('system'), ('webmaster'), ('photographer'),
    ('photographers'), ('runner'), ('runners');
