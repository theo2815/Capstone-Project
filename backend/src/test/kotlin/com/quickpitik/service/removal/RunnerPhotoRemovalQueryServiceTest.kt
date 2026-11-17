package com.quickpitik.service.removal

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito.*
import org.springframework.data.domain.Pageable
import java.time.Instant
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class RunnerPhotoRemovalQueryServiceTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val history = mock(PhotoRemovalHistoryRepository::class.java)
    private val service = RunnerPhotoRemovalQueryService(requests, items, history)
    private val runnerId = UUID.randomUUID()
    private val t0 = Instant.parse("2026-11-12T01:00:00Z")
    private val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Cebu run",
        requesterUserId = runnerId, requesterContact = "runner@test.local", supportReference = "Runner request",
        requesterReason = "Private reason", publicToOwnerReason = "p", creatorId = runnerId, creationCommandId = UUID.randomUUID(),
        source = RemovalRequestSource.RUNNER, status = RemovalRequestStatus.OPEN, livenessBand = RunnerLivenessBand.PASS)

    private fun <T> anyArg(): T = any()

    @Test fun `another runner's request is not found`() {
        val ex = assertFailsWith<ApiException> { service.detail(UUID.randomUUID(), request.id) }
        assertEquals(ErrorCodes.NOT_FOUND, ex.code)
    }

    @Test fun `a support case is not found`() {
        // The lookup is scoped to source RUNNER, so a support case with this requester is never returned.
        assertFailsWith<ApiException> { service.detail(runnerId, request.id) }
        verify(requests).findByIdAndRequesterUserIdAndSource(request.id, runnerId, RemovalRequestSource.RUNNER)
        verify(requests, never()).findById(anyArg() ?: request.id)
    }

    @Test fun `the timeline drops admin notes and actor ids`() {
        val adminId = UUID.randomUUID()
        val verify = UUID.randomUUID()
        fun row(action: String, actor: RemovalActorType, at: Long, command: UUID = UUID.randomUUID(), meta: Map<String, String> = emptyMap()) =
            PhotoRemovalHistory(UUID.randomUUID(), request.id, UUID.randomUUID(), null, UUID.randomUUID(), actor, adminId, action,
                null, null, "Secret admin note", t0.plusSeconds(at), command, meta)
        `when`(requests.findByIdAndRequesterUserIdAndSource(request.id, runnerId, RemovalRequestSource.RUNNER)).thenReturn(request)
        `when`(history.findByRequestIdOrderByOccurredAtAscIdAsc(request.id, Pageable.unpaged())).thenReturn(listOf(
            row(RemovalHistoryActions.CREATED, RemovalActorType.RUNNER, 0),
            row(RemovalHistoryActions.VERIFIED, RemovalActorType.SYSTEM, 1, verify), // two photos hidden by one command: one step
            row(RemovalHistoryActions.VERIFIED, RemovalActorType.SYSTEM, 2, verify),
            row(RemovalHistoryActions.NOTICE_STORED, RemovalActorType.SYSTEM, 3),     // not on the allow-list
            row(RemovalHistoryActions.ADMIN_DECIDED, RemovalActorType.ADMIN, 4, meta = mapOf("decision" to "MORE_EVIDENCE", "decisionLogId" to "x")),
            row(RemovalHistoryActions.CANCEL_REQUESTED, RemovalActorType.ADMIN, 5),   // an admin cancel is not a runner withdraw
        ))
        `when`(items.countStatesByRequestIds(listOf(request.id))).thenReturn(listOf(
            RequestStateCount(request.id, RemovalItemState.VERIFIED, RemovalDecision.MORE_EVIDENCE, 2)))

        val detail = service.detail(runnerId, request.id)

        assertEquals(listOf("SUBMITTED" to t0, "PHOTOS_HIDDEN" to t0.plusSeconds(1), "MORE_INFO_NEEDED" to t0.plusSeconds(4)),
            detail.timeline.map { it.step to it.at })
        assertEquals("ADDITIONAL_VERIFICATION_REQUIRED", detail.status)
        assertEquals(2, detail.counts.hidden)
        val json = jacksonObjectMapper().findAndRegisterModules().writeValueAsString(detail)
        for (secret in listOf("Secret admin note", adminId.toString(), "Private reason", "runner@test.local", "decisionLogId"))
            assertFalse(secret in json, "leaked $secret")
    }
}
