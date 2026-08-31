-- V40: drop the orphaned blur_score column from photos.
-- Blur detection is desktop-only (BatchMyPhotos culls before upload); the web/mobile
-- upload path never computed or wrote blur_score, so the column (added in V3) has always
-- been NULL and is mapped by no JPA entity. See root CLAUDE.md rule 6 / .claude/rules/ai-api-boundary.md.
ALTER TABLE photos DROP COLUMN IF EXISTS blur_score;
