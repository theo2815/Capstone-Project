-- V25: drop the legacy events.banner_url column.
--
-- V13 moved admin-uploaded covers to object storage (events.cover_s3_key)
-- and left banner_url in place as a read fallback for "V2 seed rows that
-- ship pre-populated URLs and any caller that ever set a remote URL
-- directly." Neither case ever materialised:
--
--   - The four V2 seed rows all inserted banner_url = NULL, and V14
--     soft-deleted them anyway.
--   - The only writer in the codebase is AdminEventService.create, which
--     hardcodes bannerUrl = null. No endpoint accepts a remote URL.
--
-- So EventDtoMapper.resolveBannerUrl's fallback branch could never fire —
-- it returned NULL on every row that reached it. Dropping the column and
-- collapsing the mapper to cover_s3_key removes a dead code path rather
-- than a working one.
--
-- Wire contract unchanged: the DTO field `bannerUrl` stays on EventDto /
-- EventDetailDto / AdminListEventDto / SavedEventSummaryDto / the
-- photographer event DTOs, still resolved from cover_s3_key. Website and
-- mobile see identical JSON.
--
-- Fresh DBs are safe: V2 creates the column and seeds it before V25 runs.

ALTER TABLE events
    DROP COLUMN banner_url;
