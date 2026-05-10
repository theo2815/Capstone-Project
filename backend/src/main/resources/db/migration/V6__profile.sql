-- V6: profile (avatar storage on users + user_selfies for face library).
--
--   users.avatar_s3_key  : S3/local key for the canonical avatar. avatar_url stays
--                          in the schema for legacy reads but isn't written by PR
--                          6+ — UserDto.avatarUrl is presigned from avatar_s3_key
--                          on every read so no URL ever expires in the row.
--
--   user_selfies         : up to 5 selfie refs per user (cap enforced in
--                          SelfieService). is_primary uniqueness is enforced by a
--                          partial unique index — at most one TRUE per user_id;
--                          delete promotes the most-recent remaining selfie in
--                          the same TX.
--
--   pgvector embedding column is intentionally omitted for v1 (the extension
--   isn't wired into the dev container yet). It can be added in a later
--   migration when faces/search needs the embedding cached server-side; for now
--   the ai-api stores the embedding under person_id = the runner's user_id.

ALTER TABLE users ADD COLUMN avatar_s3_key VARCHAR(512);

CREATE TABLE user_selfies (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID            NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    s3_key          VARCHAR(512)    NOT NULL,
    is_primary      BOOLEAN         NOT NULL DEFAULT FALSE,
    quality_score   NUMERIC(5, 4)   NOT NULL DEFAULT 0,
    uploaded_at     TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE INDEX idx_user_selfies_user_uploaded
    ON user_selfies (user_id, uploaded_at DESC);

CREATE UNIQUE INDEX uq_user_selfies_primary_per_user
    ON user_selfies (user_id) WHERE is_primary;
