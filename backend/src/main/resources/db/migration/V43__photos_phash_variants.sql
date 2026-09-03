-- V43: fingerprints of the preview frame BEFORE the watermark was drawn —
-- the full frame and its middle 60% crop — so POST /api/v1/public/photos/verify
-- still attributes a copy that was cleaned or cropped to the runner. Written
-- with `phash` at the LIVE flip; NULL on older rows (the clean render no
-- longer exists for them), and the verify query treats NULL as distance 64.
ALTER TABLE photos ADD COLUMN phash_clean BIGINT, ADD COLUMN phash_centre BIGINT;
