package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.Photo
import com.quickpitik.entity.RunnerLivenessBand
import com.quickpitik.entity.User
import com.quickpitik.service.removal.CandidatePhoto
import com.quickpitik.service.removal.PhotoRemovalService
import com.quickpitik.service.removal.RunnerCaseResult
import com.quickpitik.service.removal.RunnerDraft
import com.quickpitik.service.removal.RunnerDraftCommand
import com.quickpitik.service.removal.RunnerMatchOutcome
import java.math.BigDecimal
import java.time.Instant
import java.util.UUID

// Drives RD-4's two submit transactions directly, with a hand-built match outcome (never the route or AWS).
internal fun PhotoRemovalService.openRunner(runner: User, event: Event, band: RunnerLivenessBand = RunnerLivenessBand.PASS): RunnerDraft =
    openRunnerDraft(RunnerDraftCommand(UUID.randomUUID(), runner.id, runner.email, event, UUID.randomUUID(), "PRIVACY", null,
        "1.3", Instant.now(), "sess-${UUID.randomUUID()}", band, BigDecimal("97.50"), "image/jpeg"))

internal fun matched(vararg photos: Pair<Photo, Double>, emailVerified: Boolean = true) = RunnerMatchOutcome(false,
    photos.map { (photo, score) -> CandidatePhoto(photo.id, photo.photographerId, score, 1) }, false, emailVerified, false)

internal fun PhotoRemovalService.runnerCase(runner: User, event: Event, outcome: RunnerMatchOutcome,
    band: RunnerLivenessBand = RunnerLivenessBand.PASS): Pair<UUID, RunnerCaseResult> {
    val draft = openRunner(runner, event, band)
    return draft.requestId to completeRunnerCase(draft.requestId, outcome)
}
