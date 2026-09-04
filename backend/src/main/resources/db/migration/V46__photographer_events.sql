-- V46: photographer-owned events (2026-09-04)
--   Photographers may create their own events. Every one is reviewed by an
--   admin before it goes live (uploads stay closed until then), can be PUBLIC
--   (listed on /events) or UNLISTED (link-only), and is either PAID (normal
--   protection + commission) or FREE (no QuickPitik mark, optional own logo,
--   originals downloadable by anyone). Admin-created events keep the column
--   defaults and behave exactly as before.
--
--   created_by       : owner photographer; NULL = platform/admin event.
--   visibility       : PUBLIC | UNLISTED — only EventRepository.search filters
--                      on it; every other lookup (slug, coverage, cart, orders,
--                      saved events, alerts) must keep working by link.
--   pricing_mode     : PAID | FREE. FREE forces price_per_photo = 0 and a
--                      watermark_policy of OWN or NONE.
--   watermark_policy : PLATFORM (QuickPitik layers + logo, the pre-V46
--                      behaviour) | OWN (logo only) | NONE (plain frame).
--   review_status    : PENDING (submitted, status DRAFT) | APPROVED (live) |
--                      REJECTED (status DRAFT, review_note explains) |
--                      CHANGE_PENDING (live on the OLD pricing trio while the
--                      request in pending_change awaits an admin).
--   pending_change   : { pricingMode, pricePerPhoto, watermarkPolicy,
--                      requestedAt } — non-null only in CHANGE_PENDING. The
--                      live event is never edited by the photographer's
--                      request; an admin approval applies it.
--   review_note      : the last rejection reason (initial or change).
--
--   photographer_messages.kind gains the four event decisions.

ALTER TABLE events
    ADD COLUMN created_by       UUID REFERENCES users (id) ON DELETE SET NULL,
    ADD COLUMN visibility       VARCHAR(10) NOT NULL DEFAULT 'PUBLIC'
                                CHECK (visibility IN ('PUBLIC', 'UNLISTED')),
    ADD COLUMN pricing_mode     VARCHAR(10) NOT NULL DEFAULT 'PAID'
                                CHECK (pricing_mode IN ('PAID', 'FREE')),
    ADD COLUMN watermark_policy VARCHAR(10) NOT NULL DEFAULT 'PLATFORM'
                                CHECK (watermark_policy IN ('PLATFORM', 'OWN', 'NONE')),
    ADD COLUMN review_status    VARCHAR(16) NOT NULL DEFAULT 'APPROVED'
                                CHECK (review_status IN ('PENDING', 'APPROVED', 'REJECTED', 'CHANGE_PENDING')),
    ADD COLUMN pending_change   JSONB,
    ADD COLUMN review_note      VARCHAR(500),
    ADD COLUMN reviewed_at      TIMESTAMPTZ,
    ADD COLUMN reviewed_by      UUID REFERENCES users (id) ON DELETE SET NULL;

CREATE INDEX idx_events_review_queue
    ON events (created_at)
    WHERE review_status IN ('PENDING', 'CHANGE_PENDING') AND deleted_at IS NULL;

CREATE INDEX idx_events_created_by
    ON events (created_by)
    WHERE created_by IS NOT NULL;

ALTER TABLE photographer_messages
    DROP CONSTRAINT photographer_messages_kind_check;

ALTER TABLE photographer_messages
    ADD CONSTRAINT photographer_messages_kind_check CHECK (
        kind IN (
            'verification_approved',
            'verification_rejected',
            'verification_reset',
            'suspended',
            'unsuspended',
            'force_edit',
            'dispute_resolved',
            'dispute_denied',
            'dispute_escalated',
            'payout_approved',
            'payout_held',
            'payout_paid',
            'payout_report_acknowledged',
            'payout_report_resolved',
            'admin_message',
            'event_approved',
            'event_rejected',
            'event_change_approved',
            'event_change_rejected'
        )
    );
