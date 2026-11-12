package com.quickpitik.service.removal

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mockito.*
import org.springframework.context.ApplicationEventPublisher
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionDefinition
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class RunnerWithdrawTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val commands = mock(PhotoRemovalCommandRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val availability = mock(PhotoAvailabilityService::class.java)
    private val evidence = mock(PhotoRemovalEvidenceService::class.java)
    private val recorder = mock(PhotoRemovalHistoryRecorder::class.java)
    private val publisher = mock(ApplicationEventPublisher::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val mapper = jacksonObjectMapper().findAndRegisterModules()
    private val now = Instant.parse("2026-11-12T02:00:00Z")
    private val runnerId = UUID.randomUUID()
    private val commandId = UUID.randomUUID()
    private val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Cebu run",
        requesterUserId = runnerId, requesterContact = "runner@test.local", supportReference = "Runner request",
        requesterReason = "r", publicToOwnerReason = "p", creatorId = runnerId, creationCommandId = UUID.randomUUID(),
        source = RemovalRequestSource.RUNNER, status = RemovalRequestStatus.OPEN, livenessBand = RunnerLivenessBand.PASS)
    private val hidden = PhotoRemovalItem(requestId = request.id, photoIdSnapshot = UUID.randomUUID(), photoId = UUID.randomUUID(),
        state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.SELFIE)
    private var receipt: PhotoRemovalCommand? = null

    private val service = PhotoRemovalService(requests, commands, mock(EventRepository::class.java), users, items,
        mock(PhotoRepository::class.java), mock(PhotoRemovalReviewRepository::class.java), mock(PhotoRemovalNotifier::class.java), availability,
        evidence, mock(PhotoRemovalQueryService::class.java), recorder, PhotoRemovalTransitions(), TransactionTemplate(manager), mapper,
        Clock.fixed(now, ZoneOffset.UTC), publisher, mock(com.quickpitik.service.photos.PhotoErasureService::class.java),
        mock(PhotoErasureJobRepository::class.java), mock(com.quickpitik.service.orders.OrderPhotoRemovalService::class.java),
        mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java), mock(PhotoRemovalHistoryRepository::class.java),
        com.quickpitik.config.PhotoRemovalProperties(), mock(RunnerAutoHidePolicy::class.java))

    init {
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition())).thenReturn(SimpleTransactionStatus())
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
        `when`(items.findByRequestIdOrderByIdAsc(request.id)).thenReturn(listOf(hidden))
        `when`(commands.findByCommandId(commandId)).thenAnswer { receipt }
        `when`(commands.insertClaim(eqId(commandId), eqId(request.id), anyString(), eqId(runnerId), anyString(), isNull(), anyInstant()))
            .thenAnswer { inv ->
                if (receipt != null) 0 else {
                    receipt = PhotoRemovalCommand(commandId, request.id, RemovalCommandOperation.valueOf(inv.getArgument(2)), runnerId,
                        inv.getArgument(4), claimedAt = now)
                    1
                }
            }
    }

    private fun eqId(id: UUID): UUID = eq(id) ?: id
    private fun anyInstant(): Instant = org.mockito.ArgumentMatchers.any() ?: now

    @Test fun `withdraw after authorization is refused`() {
        `when`(items.findByRequestIdOrderByIdAsc(request.id)).thenReturn(listOf(hidden,
            PhotoRemovalItem(requestId = request.id, photoIdSnapshot = UUID.randomUUID(), state = RemovalItemState.REMOVAL_AUTHORIZED,
                matchSource = RemovalMatchSource.SELFIE, adminDecision = RemovalDecision.REMOVE)))
        val ex = assertFailsWith<ApiException> { service.withdraw(runnerId, request.id, commandId, 0) }
        assertEquals(409 to ErrorCodes.CONFLICT, ex.status.value() to ex.code)
        assertEquals(null, receipt)
        assertEquals(RemovalItemState.VERIFIED, hidden.state)
        verifyNoInteractions(evidence, availability)
    }

    @Test fun `a stale version or another runner's case is refused`() {
        assertEquals(ErrorCodes.CONFLICT, assertFailsWith<ApiException> { service.withdraw(runnerId, request.id, commandId, 7) }.code)
        assertEquals(404, assertFailsWith<ApiException> { service.withdraw(UUID.randomUUID(), request.id, commandId, 0) }.status.value())
    }

    @Test fun `same idempotency key twice withdraws once`() {
        val first = service.withdraw(runnerId, request.id, commandId, 0)
        val second = service.withdraw(runnerId, request.id, commandId, 0)
        assertEquals(first, second)
        assertEquals("WITHDRAWN", first.status)
        assertFalse(first.canWithdraw)
        verify(commands, times(1)).insertClaim(eqId(commandId), eqId(request.id), anyString(), eqId(runnerId), anyString(), isNull(), anyInstant())
        verify(recorder, times(1)).record(org.mockito.ArgumentMatchers.any() ?: RemovalHistoryEvent(request.id,
            actorType = RemovalActorType.RUNNER, action = "x", occurredAt = now, commandId = commandId))
        assertEquals(RemovalCommandOperation.CANCEL, receipt!!.operation)
    }

    @Test fun `withdraw never calls the admin check`() {
        service.withdraw(runnerId, request.id, commandId, 0)
        verifyNoInteractions(users)
    }

    @Test fun `a withdrawn case still in RESOLVING lets the runner file again`() {
        service.withdraw(runnerId, request.id, commandId, 0)
        // RESOLVING with close_reason WITHDRAWN is outside RUNNER_OPEN_SQL (RD-1's open index), so the slot is free at once.
        assertEquals(RemovalRequestStatus.RESOLVING to RunnerRemovalCloseReason.WITHDRAWN, request.status to request.closeReason)
        assertEquals(RemovalItemState.EXCLUDED, hidden.state)
        verify(availability).refreshRemovalState(setOf(hidden.photoId!!))
    }

    @Test fun `withdraw purges the reference image after commit`() {
        service.withdraw(runnerId, request.id, commandId, 0)
        val order = inOrder(manager, evidence)
        order.verify(manager).commit(org.mockito.ArgumentMatchers.any())
        order.verify(evidence).purgeResolvedEvidence(request.id)
    }
}
