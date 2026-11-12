package com.quickpitik.service.removal

import com.quickpitik.entity.*
import com.quickpitik.service.removal.RunnerRemovalStatus.*
import org.junit.jupiter.api.Test
import java.util.UUID
import kotlin.test.assertEquals

class RunnerRemovalStatusTest {
    private fun request(status: RemovalRequestStatus = RemovalRequestStatus.OPEN, close: RunnerRemovalCloseReason? = null,
                        band: RunnerLivenessBand = RunnerLivenessBand.PASS, flags: List<String> = emptyList()) =
        PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Run", requesterContact = "r",
            supportReference = "Runner request", requesterReason = "r", publicToOwnerReason = "p", creatorId = UUID.randomUUID(),
            creationCommandId = UUID.randomUUID(), source = RemovalRequestSource.RUNNER, status = status, closeReason = close,
            livenessBand = band, riskFlags = flags.toMutableList())

    private fun item(of: PhotoRemovalRequest, state: RemovalItemState, decision: RemovalDecision? = null) =
        PhotoRemovalItem(requestId = of.id, photoIdSnapshot = UUID.randomUUID(), state = state,
            matchSource = RemovalMatchSource.SELFIE, adminDecision = decision)

    private fun of(r: PhotoRemovalRequest, vararg items: Pair<RemovalItemState, RemovalDecision?>) =
        RunnerRemovalStatus.of(RunnerStatusView(r, items.map { item(r, it.first, it.second) }.stateCounts()))

    @Test fun `one row per rule`() {
        val s = RemovalItemState.entries.associateBy { it }
        assertEquals(WITHDRAWN, of(request(close = RunnerRemovalCloseReason.WITHDRAWN)))
        assertEquals(NO_MATCHES_FOUND, of(request(RemovalRequestStatus.RESOLVING, RunnerRemovalCloseReason.NO_MATCHES)))
        assertEquals(PROCESSING, of(request(RemovalRequestStatus.DRAFT, flags = listOf(RunnerRiskFlag.MATCHING_UNAVAILABLE.name))))
        assertEquals(ADDITIONAL_VERIFICATION_REQUIRED, of(request(), s[RemovalItemState.VERIFIED]!! to RemovalDecision.MORE_EVIDENCE))
        assertEquals(ADDITIONAL_VERIFICATION_REQUIRED, of(request(band = RunnerLivenessBand.UNCERTAIN), RemovalItemState.CANDIDATE to null))
        assertEquals(PENDING_ADMIN_REVIEW, of(request(), RemovalItemState.CANDIDATE to null))
        assertEquals(PENDING_ADMIN_REVIEW, of(request(), RemovalItemState.VERIFIED to null))
        assertEquals(APPROVED, of(request(), RemovalItemState.REMOVAL_AUTHORIZED to RemovalDecision.REMOVE))
        assertEquals(COMPLETED, of(request(RemovalRequestStatus.RESOLVED), RemovalItemState.REMOVED to RemovalDecision.REMOVE))
        assertEquals(REJECTED, of(request(RemovalRequestStatus.RESOLVED), RemovalItemState.RETAINED to RemovalDecision.RETAIN))
        assertEquals(REJECTED, of(request(RemovalRequestStatus.CANCELLED), RemovalItemState.EXCLUDED to null))
    }

    @Test fun `withdrawn beats suggested items`() =
        assertEquals(WITHDRAWN, of(request(close = RunnerRemovalCloseReason.WITHDRAWN), RemovalItemState.CANDIDATE to null))

    @Test fun `MORE_EVIDENCE beats pending`() = assertEquals(ADDITIONAL_VERIFICATION_REQUIRED,
        of(request(), RemovalItemState.CANDIDATE to null, RemovalItemState.VERIFIED to RemovalDecision.MORE_EVIDENCE))

    @Test fun `one REMOVED and one RETAINED is COMPLETED`() = assertEquals(COMPLETED, of(request(RemovalRequestStatus.RESOLVED),
        RemovalItemState.REMOVED to RemovalDecision.REMOVE, RemovalItemState.RETAINED to RemovalDecision.RETAIN))

    @Test fun `an uncertain check with something already authorized is not held for more verification`() = assertEquals(APPROVED,
        of(request(band = RunnerLivenessBand.UNCERTAIN), RemovalItemState.REMOVAL_AUTHORIZED to RemovalDecision.REMOVE))
}
