package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.websocket.AdminInboxEvent
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.context.ApplicationEventPublisher
import org.springframework.data.domain.Pageable
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionDefinition
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.*

class PhotoRemovalSweepTest {
    private val deadline = Instant.parse("2026-10-08T00:00:00Z")
    private val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Event",
        requesterContact = "private contact", supportReference = "private ticket", requesterReason = "private reason",
        publicToOwnerReason = "Privacy removal request", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(), status = RemovalRequestStatus.OPEN)
    private val review = PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = UUID.randomUUID(), notifiedAt = deadline.minusSeconds(72 * 3600), deadline = deadline)
    private val item = PhotoRemovalItem(requestId = request.id, photoIdSnapshot = UUID.randomUUID(), ownerIdSnapshot = review.ownerIdSnapshot,
        reviewId = review.id, state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL)
    private val requests = mock(PhotoRemovalRequestRepository::class.java) { invocation ->
        if (invocation.method.name == "findDueActionRequestIds") {
            assertEquals(1, invocation.getArgument<Pageable>(1).pageSize)
            listOf(request.id)
        } else org.mockito.Answers.RETURNS_DEFAULTS.answer(invocation)
    }
    private val reviews = mock(PhotoRemovalReviewRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val auditRepository = mock(PhotoRemovalHistoryRepository::class.java)
    private val recorder = mock(PhotoRemovalHistoryRecorder::class.java)
    private val publisher = mock(ApplicationEventPublisher::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val tx = TransactionTemplate(manager)
    private val audit = mutableListOf<RemovalHistoryEvent>()
    private val events = mutableListOf<AdminInboxEvent>()
    init {
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition())).thenReturn(SimpleTransactionStatus())
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
        `when`(reviews.findByRequestIdOrderByIdAsc(request.id)).thenReturn(listOf(review))
        `when`(items.findByRequestIdOrderByIdAsc(request.id)).thenReturn(listOf(item))
        doAnswer { audit += it.getArgument<RemovalHistoryEvent>(0); null }.`when`(recorder).record(org.mockito.ArgumentMatchers.any() ?: event())
        doAnswer { events += it.getArgument<AdminInboxEvent>(0); null }.`when`(publisher).publishEvent(org.mockito.ArgumentMatchers.any(Object::class.java) ?: Object())
    }

    // A >= / > boundary error, consent synthesis, or repeated audit breaks this test.
    @Test fun silenceDoesNotBecomeApproval() {
        assertEquals(0, sweep(deadline.minusNanos(1)))
        assertEquals(1, sweep(deadline))
        assertEquals(0, sweep(deadline.plusSeconds(1)))
        assertTrue(review.overdue)
        assertEquals(deadline, review.overdueAt)
        assertNull(item.ownerResponse)
        assertNull(item.adminDecision)
        assertEquals(RemovalItemState.VERIFIED, item.state)
        assertEquals(1, audit.count { it.action == "REVIEW_OVERDUE" })
        val payload = events.single().payload
        assertEquals(setOf("type", "entityId", "actorId", "occurredAt"), payload.keys)
        assertEquals("photo_removal_changed", payload["type"])
        assertEquals(request.id.toString(), payload["entityId"])
        assertNull(payload["actorId"])
    }

    @Test fun `answered or decided items do not become overdue`() {
        item.ownerResponse = PhotographerRemovalResponse.AGREE
        assertEquals(0, sweep(deadline))
        item.ownerResponse = null; item.state = RemovalItemState.RETAINED
        assertEquals(0, sweep(deadline))
        assertFalse(review.overdue); assertTrue(audit.isEmpty()); assertTrue(events.isEmpty())
    }

    @Test fun `cancelled case and nonpositive limit cannot escalate`() {
        assertEquals(0, sweep(deadline, 0))
        request.status = RemovalRequestStatus.CANCELLED
        assertEquals(0, sweep(deadline))
        assertFalse(review.overdue); assertTrue(audit.isEmpty())
    }

    private fun sweep(now: Instant, limit: Int = 1) = sweepService().sweepDue(now, limit)
    private fun sweepService() = PhotoRemovalSweep(requests, reviews, items, auditRepository, recorder, publisher, tx,
        Clock.fixed(deadline, ZoneOffset.UTC), PhotoRemovalProperties(sweepBatchSize = 1))

    @Test fun `scheduled sweep uses configured action budget and injected time`() {
        sweepService().sweep()
        assertTrue(review.overdue)
        assertEquals(deadline, review.overdueAt)
        assertEquals(1, audit.size)
        assertEquals(deadline, audit.single().occurredAt)
    }
    private fun event() = RemovalHistoryEvent(requestId = request.id, actorType = RemovalActorType.SYSTEM,
        action = "REVIEW_OVERDUE", occurredAt = deadline, commandId = UUID.randomUUID())
}
