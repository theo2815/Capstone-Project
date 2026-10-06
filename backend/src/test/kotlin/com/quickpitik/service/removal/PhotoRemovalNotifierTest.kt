package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.websocket.PhotographerMessageCreatedEvent
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.context.ApplicationEventPublisher
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionDefinition
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.*

class PhotoRemovalNotifierTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val reviews = mock(PhotoRemovalReviewRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val messages = mock(PhotographerMessageRepository::class.java)
    private val history = mock(PhotoRemovalHistoryRecorder::class.java)
    private val publisher = mock(ApplicationEventPublisher::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val tx = TransactionTemplate(manager)
    private val now = Instant.parse("2026-10-05T00:00:00Z")
    private val owner = User(email = "synthetic@test.local", passwordHash = "unused", name = "Owner", role = Role.PHOTOGRAPHER)
    private val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Event",
        requesterContact = "private contact", supportReference = "private ticket", requesterReason = "private reason",
        publicToOwnerReason = "Privacy removal request", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(), status = RemovalRequestStatus.OPEN)
    private val review = PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = owner.id, ownerId = owner.id)
    private val selected = PhotoRemovalItem(requestId = request.id, photoIdSnapshot = UUID.randomUUID(), ownerIdSnapshot = owner.id,
        reviewId = review.id, state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL)
    private val writer = AdminDecisionLogService(mock(AdminDecisionLogRepository::class.java), messages, publisher)
    private val notifier = PhotoRemovalNotifier(requests, reviews, items, users, messages, writer, history,
        tx, Clock.fixed(now, ZoneOffset.UTC), PhotoRemovalProperties())
    private val persisted = mutableListOf<PhotographerMessage>()
    private val events = mutableListOf<PhotographerMessageCreatedEvent>()
    private val audit = mutableListOf<RemovalHistoryEvent>()

    init {
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition())).thenReturn(SimpleTransactionStatus())
        `when`(reviews.findById(review.id)).thenReturn(Optional.of(review))
        `when`(reviews.findByIdAndRequestId(review.id, request.id)).thenReturn(review)
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
        `when`(users.findById(owner.id)).thenReturn(Optional.of(owner))
        `when`(items.findByRequestIdOrderByIdAsc(request.id)).thenReturn(listOf(selected))
        `when`(messages.save(org.mockito.ArgumentMatchers.any() ?: message())).thenAnswer { persisted += it.getArgument<PhotographerMessage>(0); persisted.last() }
        doAnswer { events += it.getArgument<PhotographerMessageCreatedEvent>(0); null }.`when`(publisher).publishEvent(org.mockito.ArgumentMatchers.any(Object::class.java) ?: Object())
        doAnswer { audit += it.getArgument<RemovalHistoryEvent>(0); null }.`when`(history).record(org.mockito.ArgumentMatchers.any() ?: historyEvent())
    }

    @Test fun `notice uses existing writer with scoped safe link and stored time`() {
        notifier.notifyReview(review.id)
        val notice = persisted.single()
        assertEquals("photo_removal_request", notice.kindWire)
        assertEquals(request.id, notice.photoRemovalRequestId)
        assertEquals("removal:review:${review.id}:requested", notice.noticeKey)
        assertTrue(notice.body.contains(selected.id.toString()))
        assertFalse(notice.body.contains(request.requesterContact))
        assertFalse(notice.body.contains(request.requesterReason))
        assertEquals(now, review.notifiedAt)
        assertEquals(Instant.parse("2026-10-08T00:00:00Z"), review.deadline)
        val wire = events.single().payload["message"] as Map<*, *>
        assertEquals(request.id.toString(), wire["photoRemovalRequestId"])
        assertEquals(notice.id.toString(), wire["id"])
        assertEquals("NOTICE_STORED", audit.single().action)
    }

    @Test fun `missing suspended and wrong role never receive review message`() {
        for (account in listOf(null, owner.apply { suspendedAt = now.atOffset(ZoneOffset.UTC) },
            User(email = "runner@test.local", passwordHash = "unused", name = "Runner", role = Role.RUNNER))) {
            review.unavailableReason = null
            `when`(users.findById(owner.id)).thenReturn(Optional.ofNullable(account))
            notifier.notifyReview(review.id)
            assertNotNull(review.unavailableReason)
            assertNull(review.notifiedAt)
            assertNull(review.deadline)
        }
        assertTrue(persisted.isEmpty())
        assertTrue(events.isEmpty())
        assertTrue(audit.all { it.action == "OWNER_UNAVAILABLE" })
    }

    @Test fun `removed inbox notice is reused without republishing or resetting deadline`() {
        val existing = message().apply { removedAt = now.atOffset(ZoneOffset.UTC) }
        `when`(messages.findByNoticeKey("removal:review:${review.id}:requested")).thenReturn(existing)
        notifier.notifyReview(review.id)
        assertEquals(existing.id, review.notifiedMessageId)
        assertEquals(existing.createdAt.toInstant(), review.notifiedAt)
        assertEquals(existing.createdAt.toInstant().plusSeconds(72 * 3600), review.deadline)
        assertTrue(persisted.isEmpty()); assertTrue(events.isEmpty())
    }

    @Test fun `closed case is not notified`() {
        request.status = RemovalRequestStatus.CANCELLED
        notifier.notifyReview(review.id)
        assertTrue(persisted.isEmpty()); assertNull(review.deadline)
    }

    @Test fun `retry metadata database failure cannot block next owner notice`() {
        val nextOwner = User(email = "next@test.local", passwordHash = "unused", name = "Next", role = Role.PHOTOGRAPHER)
        val nextReview = PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = nextOwner.id, ownerId = nextOwner.id)
        val nextItem = PhotoRemovalItem(requestId = request.id, photoIdSnapshot = UUID.randomUUID(), ownerIdSnapshot = nextOwner.id,
            reviewId = nextReview.id, state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL)
        `when`(reviews.findById(nextReview.id)).thenReturn(Optional.of(nextReview))
        `when`(reviews.findByIdAndRequestId(nextReview.id, request.id)).thenReturn(nextReview)
        `when`(users.findById(nextOwner.id)).thenReturn(Optional.of(nextOwner))
        `when`(items.findByRequestIdOrderByIdAsc(request.id)).thenReturn(listOf(selected, nextItem))
        `when`(reviews.findPendingNotices(org.mockito.ArgumentMatchers.any() ?: now,
            org.mockito.ArgumentMatchers.any() ?: org.springframework.data.domain.PageRequest.of(0, 50))).thenAnswer {
            assertEquals(now, it.getArgument<Instant>(0))
            assertEquals(50, it.getArgument<org.springframework.data.domain.Pageable>(1).pageSize)
            listOf(review, nextReview)
        }
        doAnswer {
            val notice = it.getArgument<PhotographerMessage>(0)
            if (notice.photographerId == owner.id) throw IllegalStateException("synthetic notice failure")
            persisted += notice
            notice
        }.`when`(messages).save(org.mockito.ArgumentMatchers.any() ?: message())
        var transactions = 0
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition())).thenAnswer {
            transactions++
            if (transactions == 3) throw IllegalStateException("synthetic retry metadata failure")
            SimpleTransactionStatus()
        }
        notifier.retryPending()
        assertNull(review.notifiedAt); assertNull(review.deadline)
        assertEquals(now, nextReview.notifiedAt)
        assertEquals(nextOwner.id, persisted.single().photographerId)
    }

    private fun message() = PhotographerMessage(photographerId = owner.id, kindWire = "photo_removal_request", body = "safe",
        photoRemovalRequestId = request.id, noticeKey = "removal:review:${review.id}:requested", createdAt = now.minusSeconds(60).atOffset(ZoneOffset.UTC))
    private fun historyEvent() = RemovalHistoryEvent(requestId = request.id, actorType = RemovalActorType.SYSTEM, action = "NOTICE_STORED", occurredAt = now, commandId = UUID.randomUUID())
}
