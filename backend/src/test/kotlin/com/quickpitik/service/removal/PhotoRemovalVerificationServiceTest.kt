package com.quickpitik.service.removal

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionDefinition
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.*

// Supplemental unit verification of the contract already covered by the feature RED and PostgreSQL tests.
class PhotoRemovalVerificationServiceTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val commands = mock(PhotoRemovalCommandRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val photos = mock(PhotoRepository::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val mapper = jacksonObjectMapper().findAndRegisterModules()
    private val now = Instant.parse("2026-10-05T00:00:00Z")
    private val eventId = UUID.randomUUID()
    private val request = PhotoRemovalRequest(eventId = eventId, eventIdSnapshot = eventId, eventNameSnapshot = "Synthetic run",
        requesterContact = "private contact", supportReference = "support-1", requesterReason = "private reason", publicToOwnerReason = "Privacy removal request",
        creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(), createdAt = now)
    private val command = VerifyRemovalCommand(UUID.randomUUID(), 0, "Verified support identity", "Confirmed",
        listOf(VerifyRemovalSelection(UUID.randomUUID(), RemovalMatchSource.MANUAL, "Visually confirmed")))
    private val service = PhotoRemovalService(requests, commands, mock(EventRepository::class.java), mock(UserRepository::class.java), items,
        photos, mock(PhotoRemovalReviewRepository::class.java), mock(PhotoRemovalNotifier::class.java), mock(PhotoAvailabilityService::class.java),
        mock(PhotoRemovalEvidenceService::class.java), mock(PhotoRemovalQueryService::class.java), mock(PhotoRemovalHistoryRecorder::class.java),
        PhotoRemovalTransitions(), TransactionTemplate(manager), mapper, Clock.fixed(now, ZoneOffset.UTC), mock(org.springframework.context.ApplicationEventPublisher::class.java))

    init {
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition())).thenReturn(SimpleTransactionStatus())
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
    }

    @Test fun `stale verification rejects before selection or persistence`() {
        request.version = 3
        assertEquals(409, assertFailsWith<ApiException> { service.verify(request.creatorId, request.id, command) }.status.value())
        assertEquals(RemovalRequestStatus.DRAFT, request.status)
        assertNull(request.verifiedAt)
        assertTrue(mockingDetails(items).invocations.isEmpty())
        assertTrue(mockingDetails(photos).invocations.isEmpty())
    }

    @Test fun `missing selected photo does not open or verify case`() {
        `when`(photos.findAllByIdForUpdate(listOf(command.selectedPhotos.single().photoId))).thenReturn(emptyList())
        assertEquals(400, assertFailsWith<ApiException> { service.verify(request.creatorId, request.id, command) }.status.value())
        assertEquals(RemovalRequestStatus.DRAFT, request.status)
        assertNull(request.verifiedAt)
        assertTrue(mockingDetails(items).invocations.isEmpty())
    }

    @Test fun `successful receipt replays original result after aggregate version changes`() {
        val original = request.toAdminDto(emptyList(), emptyList())
        val receipt = PhotoRemovalCommand(command.commandId, request.id, RemovalCommandOperation.VERIFY, request.creatorId,
            commandHash(mapper, command), state = RemovalCommandState.SUCCEEDED, result = mapper.valueToTree(original), claimedAt = now)
        `when`(commands.findByCommandId(command.commandId)).thenReturn(receipt)
        request.status = RemovalRequestStatus.RESOLVED
        request.version = 10
        assertEquals(original, service.verify(request.creatorId, request.id, command))
        assertEquals(RemovalRequestStatus.RESOLVED, request.status)
        assertTrue(mockingDetails(items).invocations.isEmpty())
        assertTrue(mockingDetails(photos).invocations.isEmpty())
    }
}
