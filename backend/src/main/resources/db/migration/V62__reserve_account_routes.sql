-- Flat photographer URLs must not collide with application routes.
INSERT INTO reserved_handles (handle) VALUES
    ('reset-password'), ('verify'), ('verify-email'), ('confirm-email-change')
ON CONFLICT (handle) DO NOTHING;

-- Preserve existing ownership; operators must arrange a replacement handle
-- with any affected photographer before publishing the conflicting route.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM photographer_settings p
        JOIN reserved_handles r ON r.handle = lower(p.handle)
    ) THEN
        RAISE WARNING 'Reserved portfolio handles are already claimed. Review photographer_settings joined to reserved_handles and arrange replacement handles with the owners.';
    END IF;
END $$;
