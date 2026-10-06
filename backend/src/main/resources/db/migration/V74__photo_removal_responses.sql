ALTER TABLE photo_removal_commands DROP CONSTRAINT photo_removal_commands_operation_check;
ALTER TABLE photo_removal_commands ADD CONSTRAINT photo_removal_commands_operation_check
    CHECK (operation IN ('CREATE', 'REPLACE_EVIDENCE', 'CANCEL', 'MATCH_CANDIDATES', 'VERIFY', 'RESPOND'));
