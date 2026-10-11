-- Refresh-token rotation grace (desktop upload review 2026-10-11, F1). When
-- the answer to a rotation is lost in transit the client re-presents the
-- parent token; for a short window after rotation the parent is accepted once
-- more, the child nobody received is revoked and a new child issued.
-- replaced_by_id links a revoked parent to the child it was rotated into;
-- tokens revoked by sign-out or password change never get one.
ALTER TABLE refresh_tokens ADD COLUMN replaced_by_id UUID;
