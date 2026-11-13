-- AA review fixes (AF-1): the one migration of the branch, so AF-5, AF-6 and AF-7 need none.
-- Additive with constant defaults (no table rewrite), no backfill: existing rows start at 0 tries,
-- not on hold, not mailed.

-- R-8 (AF-5): failed automatic identity tries; after a few, a later pass goes to an admin.
ALTER TABLE photographer_seller_details
    ADD COLUMN failed_auto_checks INTEGER NOT NULL DEFAULT 0;

-- R-10 (AF-6): a profile an admin reset or rejected waits for an admin on resubmit.
ALTER TABLE photographer_settings
    ADD COLUMN verification_admin_hold BOOLEAN NOT NULL DEFAULT FALSE;

-- R-7 (AF-7): the identity-notice start date this user was already mailed for.
ALTER TABLE users
    ADD COLUMN identity_notice_sent_for DATE;
