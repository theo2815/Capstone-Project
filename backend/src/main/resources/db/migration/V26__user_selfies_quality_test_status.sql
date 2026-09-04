-- V26: user_selfies.quality_test_status — separates "ai-api never looked at this
-- selfie" from "ai-api looked and it passed".
--
-- When AI_API_ENABLED=false (the application.yml default) SelfieService skips the
-- quality gate entirely and persists the selfie with quality_score = 0. Nothing
-- on the wire said whether a 0 meant "untested" or "tested and scored badly", so
-- the runner's library couldn't banner the difference or prompt a re-upload once
-- AI came online. It WAS derivable — a selfie that passes always scores >= 0.6000,
-- so 0 implies untested — but that's a magic value two client codebases would
-- each have to know about.
--
-- Two states only. There is deliberately no 'rejected': SelfieService.qualityGate
-- throws before storage or save, so a rejected selfie never gets a row to carry a
-- status. Rejections are surfaced synchronously as 4xx SELFIE_REJECTED instead.
--
-- Backfill: any existing row scoring above zero got there by passing the gate.

ALTER TABLE user_selfies
    ADD COLUMN quality_test_status VARCHAR(16) NOT NULL DEFAULT 'untested';

UPDATE user_selfies
    SET quality_test_status = 'passed'
    WHERE quality_score > 0;

ALTER TABLE user_selfies
    ADD CONSTRAINT user_selfies_quality_test_status_check CHECK (
        quality_test_status IN (
            'untested',
            'passed'
        )
    );
