-- Photographer watermark placement: position (independent per orientation),
-- opacity, and size. Before this, WatermarkService baked every logo at a fixed
-- bottom-right corner, 15% of photo width, 70% opacity. These six columns make
-- those three knobs photographer-controllable; only position is split by
-- orientation (landscape vs portrait), size + opacity are shared.
--
-- Every default reproduces the historical hardcoded mark exactly, so existing
-- photographers' output is unchanged until they customize:
--   opacity 0.70 · scale 0.15 (fraction of photo width) · position (1,1) =
--   the padded bottom-right corner. Position values are normalized free-space
--   fractions in [0,1]; the compositor maps them to pixels per photo.
ALTER TABLE photographer_settings
    ADD COLUMN watermark_opacity         double precision NOT NULL DEFAULT 0.70,
    ADD COLUMN watermark_scale           double precision NOT NULL DEFAULT 0.15,
    ADD COLUMN watermark_pos_landscape_x double precision NOT NULL DEFAULT 1.0,
    ADD COLUMN watermark_pos_landscape_y double precision NOT NULL DEFAULT 1.0,
    ADD COLUMN watermark_pos_portrait_x  double precision NOT NULL DEFAULT 1.0,
    ADD COLUMN watermark_pos_portrait_y  double precision NOT NULL DEFAULT 1.0;
