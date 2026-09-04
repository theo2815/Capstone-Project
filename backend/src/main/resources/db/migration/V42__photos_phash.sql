-- V42: 64-bit perceptual hash of the marked preview (watermark.jpg), the
-- fingerprint POST /api/v1/public/photos/verify matches a screenshot against.
-- NULL until PhotoWatermarkService computes it at the LIVE flip (new uploads)
-- or PhotoWatermarkTrigger.backfillPhash fills it in (pre-V42 rows).
-- ponytail: no index — Hamming search is a scan by nature; revisit past ~1M rows.
ALTER TABLE photos ADD COLUMN phash BIGINT;
