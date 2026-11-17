package com.quickpitik.service.removal

import com.quickpitik.entity.*
import com.quickpitik.repository.RequestStateCount

/** What a runner sees of their deletion request (RD-5). Built from [RunnerStatusView] by [of]; first match wins. */
enum class RunnerRemovalStatus {
    PROCESSING, PENDING_ADMIN_REVIEW, ADDITIONAL_VERIFICATION_REQUIRED, NO_MATCHES_FOUND, APPROVED, REJECTED, COMPLETED, WITHDRAWN;

    companion object {
        fun of(view: RunnerStatusView): RunnerRemovalStatus {
            val r = view.request
            fun has(state: RemovalItemState, decided: (RemovalDecision?) -> Boolean = { true }) =
                view.counts.any { it.state == state && it.count > 0 && decided(it.decision) }
            val undecided = has(RemovalItemState.CANDIDATE) || has(RemovalItemState.VERIFIED) { it == null }
            val settled = has(RemovalItemState.REMOVAL_AUTHORIZED) || has(RemovalItemState.REMOVED)
            return when {
                r.closeReason == RunnerRemovalCloseReason.WITHDRAWN -> WITHDRAWN
                r.closeReason == RunnerRemovalCloseReason.NO_MATCHES -> NO_MATCHES_FOUND
                r.status == RemovalRequestStatus.DRAFT && view.counts.isEmpty() &&
                    RunnerRiskFlag.MATCHING_UNAVAILABLE.name in r.riskFlags -> PROCESSING
                has(RemovalItemState.VERIFIED) { it == RemovalDecision.MORE_EVIDENCE } ||
                    (r.livenessBand == RunnerLivenessBand.UNCERTAIN && undecided && !settled) -> ADDITIONAL_VERIFICATION_REQUIRED
                undecided -> PENDING_ADMIN_REVIEW
                has(RemovalItemState.REMOVAL_AUTHORIZED) -> APPROVED
                has(RemovalItemState.REMOVED) -> COMPLETED
                else -> REJECTED
            }
        }
    }
}

/** One case and its item counts; built in one place, [RunnerPhotoRemovalQueryService.view] (RD-9 reuses it). */
data class RunnerStatusView(val request: PhotoRemovalRequest, val counts: List<RequestStateCount>)

internal fun List<PhotoRemovalItem>.stateCounts(): List<RequestStateCount> =
    groupingBy { Triple(it.requestId, it.state, it.adminDecision) }.eachCount()
        .map { (k, n) -> RequestStateCount(k.first, k.second, k.third, n.toLong()) }

/** Admin cancel and runner withdraw: only an open case with nothing authorized or removed can close. */
internal fun removalClosable(status: RemovalRequestStatus, states: Collection<RemovalItemState>) =
    status in setOf(RemovalRequestStatus.DRAFT, RemovalRequestStatus.OPEN) &&
        states.none { it == RemovalItemState.REMOVAL_AUTHORIZED || it == RemovalItemState.REMOVED }
