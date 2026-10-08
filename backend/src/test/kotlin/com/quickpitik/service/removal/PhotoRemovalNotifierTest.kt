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
    private val historyReader = mock(PhotoRemovalHistoryRepository::class.java)
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
        tx, Clock.fixed(now, ZoneOffset.UTC), PhotoRemovalProperties(), historyReader)
    private val persisted = mutableListOf<PhotographerMessage>()
    private val events = mutableListOf<PhotographerMessageCreatedEvent>()
    private val audit = mutableListOf<RemovalHistoryEvent>()

    init {
        `when`(historyReader.findReviewNoticeVersion(org.mockito.ArgumentMatchers.any() ?: review.id)).thenReturn(null)
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

    @Test fun `cancelled case pending outcome failure retains durable backoff`() {
        request.status = RemovalRequestStatus.CANCELLED
        val pending = PhotoRemovalHistory(UUID.randomUUID(), request.id, selected.id, review.id, selected.photoIdSnapshot,
            RemovalActorType.ADMIN, UUID.randomUUID(), "ADMIN_DECIDED", "VERIFIED", "VERIFIED", "private reason", now,
            UUID.randomUUID(), mapOf("decisionVersion" to "2"))
        `when`(historyReader.findPendingOutcomes(org.mockito.ArgumentMatchers.eq(review.id) ?: review.id,
            org.mockito.ArgumentMatchers.any() ?: org.springframework.data.domain.PageRequest.of(0, 100))).thenReturn(listOf(pending))
        doThrow(IllegalStateException("synthetic outcome failure")).`when`(messages).save(org.mockito.ArgumentMatchers.any() ?: message())
        notifier.attempt(review.id)
        assertEquals(now.plus(PhotoRemovalProperties().cleanupBackoff), review.notificationNextAttemptAt)
        assertTrue(audit.isEmpty()); assertTrue(persisted.isEmpty())
    }

    @Test fun `outcome is safe scoped and unavailable owner receives only audit`() {
        review.notifiedAt = now
        val pending = PhotoRemovalHistory(UUID.randomUUID(), request.id, selected.id, review.id, selected.photoIdSnapshot,
            RemovalActorType.ADMIN, UUID.randomUUID(), "ADMIN_DECIDED", "VERIFIED", "RETAINED_RESTRICTED", "private reason", now,
            UUID.randomUUID(), mapOf("decisionVersion" to "2"))
        `when`(historyReader.findPendingOutcomes(org.mockito.ArgumentMatchers.eq(review.id) ?: review.id,
            org.mockito.ArgumentMatchers.any() ?: org.springframework.data.domain.PageRequest.of(0, 100))).thenReturn(listOf(pending))
        notifier.notifyOutcomes(review.id)
        val notice = persisted.single()
        assertTrue(notice.body.contains(selected.id.toString()))
        assertTrue(notice.body.contains("not erased")); assertFalse(notice.body.contains("private"))
        assertEquals("photo_removal_outcome", notice.kindWire)
        assertEquals("removal:review:${review.id}:outcome:2", audit.single().metadata["noticeKey"])
        `when`(users.findById(owner.id)).thenReturn(Optional.empty())
        notifier.notifyOutcomes(review.id)
        assertEquals(1, persisted.size)
        assertEquals("OWNER_UNAVAILABLE", audit.last().action)
    }

    @Test fun `removal admin log keeps item identity distinct in one command`() {
        val logs = mock(AdminDecisionLogRepository::class.java)
        val rows = mutableListOf<AdminDecisionLog>()
        `when`(logs.save(org.mockito.ArgumentMatchers.any() ?: AdminDecisionLog(adminId = owner.id, decision = "sample")))
            .thenAnswer { rows += it.getArgument<AdminDecisionLog>(0); rows.last() }
        val auditWriter = AdminDecisionLogService(logs, messages, publisher)
        val commandId = UUID.randomUUID(); val second = UUID.randomUUID()
        val firstId = auditWriter.logPhotoRemovalDecision(owner.id, request.id, selected.id, RemovalDecision.RETAIN, "Reviewed", commandId)
        val secondId = auditWriter.logPhotoRemovalDecision(owner.id, request.id, second, RemovalDecision.REMOVE, "Confirmed", commandId)
        assertNotEquals(firstId, secondId)
        assertEquals(setOf(selected.id.toString(), second.toString()), rows.map { it.meta!!["itemId"] }.toSet())
        assertTrue(rows.all { it.meta!!["commandId"] == commandId.toString() && it.adminId == owner.id })
        assertEquals(listOf("photo_removal_retain", "photo_removal_remove"), rows.map { it.decision })
    }

    @Test fun `only explicitly pending retained item starts a fresh response notice`() {
        selected.state = RemovalItemState.RETAINED_RESTRICTED
        selected.adminDecision = RemovalDecision.RETAIN_RESTRICTED
        notifier.notifyReview(review.id)
        assertTrue(persisted.isEmpty()); assertNull(review.deadline)
        `when`(items.findPendingOwnerReviewItemIds(request.id)).thenReturn(listOf(selected.id))
        notifier.notifyReview(review.id)
        assertEquals(1, persisted.size)
        assertTrue(persisted.single().body.contains(selected.id.toString()))
        assertEquals(now.plusSeconds(72 * 3600), review.deadline)
        assertEquals(RemovalDecision.RETAIN_RESTRICTED, selected.adminDecision)
    }

    private fun message() = PhotographerMessage(photographerId = owner.id, kindWire = "photo_removal_request", body = "safe",
        photoRemovalRequestId = request.id, noticeKey = "removal:review:${review.id}:requested", createdAt = now.minusSeconds(60).atOffset(ZoneOffset.UTC))
    private fun historyEvent() = RemovalHistoryEvent(requestId = request.id, actorType = RemovalActorType.SYSTEM, action = "NOTICE_STORED", occurredAt = now, commandId = UUID.randomUUID())
}
