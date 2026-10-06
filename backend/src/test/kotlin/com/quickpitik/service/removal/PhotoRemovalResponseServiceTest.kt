package com.quickpitik.service.removal

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.storage.StorageService
import com.quickpitik.websocket.AdminInboxEvent
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.context.ApplicationEventPublisher
import org.springframework.data.domain.PageImpl
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

// Supplemental unit checks for the behavior first exercised by the HTTP/PostgreSQL RED batch.
class PhotoRemovalResponseServiceTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val commands = mock(PhotoRemovalCommandRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val reviews = mock(PhotoRemovalReviewRepository::class.java)
    private val photos = mock(PhotoRepository::class.java)
    private val history = mock(PhotoRemovalHistoryRecorder::class.java)
    private val publisher = mock(ApplicationEventPublisher::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val tx = TransactionTemplate(manager)
    private val mapper = jacksonObjectMapper().findAndRegisterModules()
    private val now = Instant.parse("2026-10-05T00:00:00Z")
    private val clock = Clock.fixed(now, ZoneOffset.UTC)
    private val owner = UUID.randomUUID()
    private val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Event",
        requesterContact = "private contact", supportReference = "private ticket", requesterReason = "private reason",
        publicToOwnerReason = "Privacy removal request", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(), status = RemovalRequestStatus.OPEN)
    private val review = PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = owner,
        notifiedAt = now, deadline = Instant.parse("2026-10-08T00:00:00Z"))
    private val item = PhotoRemovalItem(requestId = request.id, photoIdSnapshot = UUID.randomUUID(), ownerIdSnapshot = owner,
        reviewId = review.id, state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL)
    private val queries = PhotoRemovalQueryService(requests, items, reviews, mock(PhotoRemovalHistoryRepository::class.java), photos,
        mock(UserRepository::class.java), commands, mock(PhotoRemovalEvidenceService::class.java), mock(PhotoSearchService::class.java),
        mock(StorageService::class.java), StorageProperties(), tx, mapper, clock)
    private val service = PhotoRemovalService(requests, commands, mock(EventRepository::class.java), mock(UserRepository::class.java), items,
        photos, reviews, mock(PhotoRemovalNotifier::class.java), mock(PhotoAvailabilityService::class.java), mock(PhotoRemovalEvidenceService::class.java),
        queries, history, PhotoRemovalTransitions(), tx, mapper, clock, publisher)
    private val receipts = mutableMapOf<UUID, PhotoRemovalCommand>()
    private val audit = mutableListOf<RemovalHistoryEvent>()
    private val events = mutableListOf<AdminInboxEvent>()
    init {
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition())).thenReturn(SimpleTransactionStatus())
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
        `when`(requests.findById(request.id)).thenReturn(Optional.of(request))
        `when`(items.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(request.id, owner)).thenReturn(listOf(item))
        `when`(reviews.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(request.id, owner)).thenReturn(listOf(review))
        `when`(requests.advanceVersion(request.id, now)).thenAnswer { request.version++; request.updatedAt = now; 1 }
        `when`(commands.findByCommandId(org.mockito.ArgumentMatchers.any() ?: UUID.randomUUID())).thenAnswer { receipts[it.getArgument(0)] }
        `when`(commands.insertClaim(org.mockito.ArgumentMatchers.any() ?: UUID.randomUUID(), org.mockito.ArgumentMatchers.eq(request.id) ?: request.id,
            org.mockito.ArgumentMatchers.eq("RESPOND") ?: "RESPOND", org.mockito.ArgumentMatchers.eq(owner) ?: owner,
            org.mockito.ArgumentMatchers.anyString() ?: "", org.mockito.ArgumentMatchers.isNull(), org.mockito.ArgumentMatchers.eq(now) ?: now)).thenAnswer {
            val id = it.getArgument<UUID>(0)
            receipts[id] = PhotoRemovalCommand(id, request.id, RemovalCommandOperation.RESPOND, owner, it.getArgument(4), claimedAt = now)
            1
        }
        doAnswer { audit += it.getArgument<RemovalHistoryEvent>(0); null }.`when`(history).record(org.mockito.ArgumentMatchers.any() ?: event())
        doAnswer { events += it.getArgument<AdminInboxEvent>(0); null }.`when`(publisher).publishEvent(org.mockito.ArgumentMatchers.any(Object::class.java) ?: Object())
    }

    @Test fun `response stores item history and original scoped receipt without deciding removal`() {
        val command = RespondToRemovalCommand(UUID.randomUUID(), 0, listOf(item.id), PhotographerRemovalResponse.OBJECT, "Please investigate")
        val first = service.respond(owner, request.id, command)
        assertEquals(PhotographerRemovalResponse.OBJECT, first.items.single().response)
        assertEquals(1L, first.version)
        assertEquals(RemovalItemState.VERIFIED, item.state)
        assertNull(item.adminDecision)
        assertEquals(review.deadline, first.reviews.single().deadline)
        assertEquals(owner, audit.single().actorId)
        assertEquals("OWNER_RESPONDED", audit.single().action)
        assertEquals(owner.toString(), events.single().payload["actorId"])
        request.version = 3; item.ownerResponse = PhotographerRemovalResponse.AGREE
        assertEquals(first, service.respond(owner, request.id, command))
        assertEquals(1, audit.size)
    }

    @Test fun `whole selection and objection validation leave every item unchanged`() {
        val valid = RespondToRemovalCommand(UUID.randomUUID(), 0, listOf(item.id), PhotographerRemovalResponse.AGREE)
        for ((code, command) in listOf(404 to valid.copy(itemIds = listOf(item.id, UUID.randomUUID())),
            400 to valid.copy(itemIds = listOf(item.id, item.id)), 400 to valid.copy(response = PhotographerRemovalResponse.OBJECT, reason = " "),
            409 to valid.copy(expectedVersion = 2))) {
            assertEquals(code, assertFailsWith<ApiException> { service.respond(owner, request.id, command) }.status.value())
        }
        assertNull(item.ownerResponse); assertTrue(receipts.isEmpty()); assertTrue(audit.isEmpty()); assertTrue(events.isEmpty())
    }

    @Test fun `more evidence remains amendable but authorization and terminal case do not`() {
        item.adminDecision = RemovalDecision.MORE_EVIDENCE
        assertEquals(PhotographerRemovalResponse.AGREE, service.respond(owner, request.id,
            RespondToRemovalCommand(UUID.randomUUID(), 0, listOf(item.id), PhotographerRemovalResponse.AGREE)).items.single().response)
        item.state = RemovalItemState.REMOVAL_AUTHORIZED
        assertEquals(409, assertFailsWith<ApiException> { service.respond(owner, request.id,
            RespondToRemovalCommand(UUID.randomUUID(), 1, listOf(item.id), PhotographerRemovalResponse.AGREE)) }.status.value())
        item.state = RemovalItemState.VERIFIED; request.status = RemovalRequestStatus.CANCELLED
        assertEquals(409, assertFailsWith<ApiException> { service.respond(owner, request.id,
            RespondToRemovalCommand(UUID.randomUUID(), 1, listOf(item.id), PhotographerRemovalResponse.AGREE)) }.status.value())
    }

    @Test fun `owner detail is a private projection and list pagination remains scoped`() {
        val detail = queries.photographerDetail(owner, request.id)
        assertEquals(listOf(item.id), detail.items.map { it.id })
        assertEquals(listOf(review.id), detail.reviews.map { it.id })
        assertEquals("Privacy removal request", detail.sanitizedReason)
        assertFalse(mapper.writeValueAsString(detail).contains("private contact"))
        assertEquals(404, assertFailsWith<ApiException> { queries.photographerDetail(UUID.randomUUID(), request.id) }.status.value())
        `when`(requests.findForPhotographer(org.mockito.ArgumentMatchers.eq(owner) ?: owner,
            org.mockito.ArgumentMatchers.isNull(), org.mockito.ArgumentMatchers.isNull(), org.mockito.ArgumentMatchers.isNull(),
            org.mockito.ArgumentMatchers.any() ?: com.quickpitik.common.OffsetLimitPageable(PaginationParams(0, 100)))).thenAnswer {
            assertEquals(100, it.getArgument<org.springframework.data.domain.Pageable>(4).pageSize)
            PageImpl(listOf(request))
        }
        val page = queries.photographerList(owner, 0, 1000)
        assertEquals(100, page.limit); assertEquals(1L, page.total)
        assertEquals(request.id, page.items.single().id)
        assertTrue(page.items.single().items.isEmpty())
    }

    @Test fun `owner snapshot cannot advertise a newly committed version for earlier item responses`() {
        var committedVersion = 0L
        `when`(requests.findById(request.id)).thenAnswer {
            Optional.of(PhotoRemovalRequest(id = request.id, eventIdSnapshot = request.eventIdSnapshot, eventNameSnapshot = "Event",
                requesterContact = "private contact", supportReference = "private ticket", requesterReason = "private reason",
                publicToOwnerReason = "Privacy removal request", creatorId = request.creatorId,
                creationCommandId = request.creationCommandId, status = RemovalRequestStatus.OPEN).apply { version = committedVersion })
        }
        `when`(items.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(request.id, owner)).thenAnswer {
            // Materialize the older item snapshot; a competing response then commits version1.
            committedVersion = 1L
            listOf(item)
        }
        val snapshot = queries.photographerDetail(owner, request.id)
        assertNull(snapshot.items.single().response)
        assertEquals(0L, snapshot.version, "Older responses must carry the earlier version so stale edits are rejected")
    }

    private fun event() = RemovalHistoryEvent(requestId = request.id, actorType = RemovalActorType.PHOTOGRAPHER,
        action = "OWNER_RESPONDED", occurredAt = now, commandId = UUID.randomUUID())
}
