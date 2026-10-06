package com.quickpitik.service.removal

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.dto.removal.SearchImage
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mockito.*
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.transaction.support.DefaultTransactionDefinition
import org.springframework.transaction.support.SimpleTransactionStatus
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.time.Clock
import java.util.UUID
import java.util.Optional
import javax.imageio.ImageIO
import kotlin.test.*

class PhotoRemovalEvidenceServiceTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val commands = mock(PhotoRemovalCommandRepository::class.java)
    private val storage = mock(StorageService::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val reviews = mock(PhotoRemovalReviewRepository::class.java)
    private val history = mock(PhotoRemovalHistoryRecorder::class.java)
    private val transactionManager = mock(PlatformTransactionManager::class.java)
    private val tx = TransactionTemplate(transactionManager)
    private val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Synthetic run",
        requesterContact = "Support contact", supportReference = "ticket-1", requesterReason = "Privacy",
        publicToOwnerReason = "Privacy request", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID())
    private val service = PhotoRemovalEvidenceService(requests, commands, items, storage, reviews, history,
        mock(PhotoRemovalHistoryRepository::class.java),
        PhotoRemovalTransitions(), tx, jacksonObjectMapper().findAndRegisterModules(), Clock.systemUTC(),
        PhotoRemovalProperties(), SimpleMeterRegistry())

    init {
        `when`(transactionManager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition()))
            .thenReturn(SimpleTransactionStatus())
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
        `when`(requests.existsById(request.id)).thenReturn(true)
        `when`(items.findByRequestIdOrderByIdAsc(request.id)).thenReturn(emptyList())
    }

    @Test fun `failedReplaceKeepsPreviousEvidence`() {
        val previousKey = "photo-removal/${request.id}/evidence/${UUID.randomUUID()}"
        request.evidenceKey = previousKey
        request.evidenceCleanupState = RemovalEvidenceCleanupState.PRESENT
        val commandId = UUID.randomUUID()
        var claimed: PhotoRemovalCommand? = null
        `when`(commands.findByCommandId(commandId)).thenAnswer { claimed }
        doAnswer {
            claimed = PhotoRemovalCommand(commandId, request.id, RemovalCommandOperation.REPLACE_EVIDENCE,
                request.creatorId, it.getArgument(4), intendedEvidenceKey = it.getArgument(5), claimedAt = it.getArgument(6))
            1
        }.`when`(commands).insertClaim(
            org.mockito.ArgumentMatchers.eq(commandId) ?: commandId,
            org.mockito.ArgumentMatchers.eq(request.id) ?: request.id,
            anyString(), org.mockito.ArgumentMatchers.eq(request.creatorId) ?: request.creatorId,
            anyString(), anyString(), org.mockito.ArgumentMatchers.any() ?: java.time.Instant.EPOCH)
        doThrow(IllegalStateException("synthetic ambiguous PUT failure"))
            .`when`(storage).put(anyString(), org.mockito.ArgumentMatchers.any(ByteArray::class.java) ?: byteArrayOf(), anyString())
        assertFailsWith<ApiException> { service.replace(request.creatorId, request.id, commandId, 0, image()) }
        val caseAfterFailedUpload = request
        assertEquals(previousKey, caseAfterFailedUpload.evidenceKey)
        assertEquals(0, caseAfterFailedUpload.evidenceGeneration)
        assertTrue("photo-removal/${request.id}/evidence/$commandId" in request.outstandingEvidenceKeys)
        assertEquals(RemovalEvidenceCleanupState.RETRY_REQUIRED, request.evidenceCleanupState)
    }

    @Test fun `cancelRetainsCleanupWorkOnStorageFailure`() {
        request.status = RemovalRequestStatus.RESOLVING
        request.evidenceKey = "photo-removal/${request.id}/evidence/${UUID.randomUUID()}"
        request.evidenceCleanupState = RemovalEvidenceCleanupState.PENDING
        doThrow(IllegalStateException("synthetic DELETE failure")).`when`(storage).delete(request.evidenceKey!!)
        val purgeResult = service.purgeResolvedEvidence(request.id)
        val caseAfterFailedEvidenceDelete = request
        assertNull(caseAfterFailedEvidenceDelete.resolvedAt)
        assertEquals(EvidenceCleanupResult.RETRY_REQUIRED, purgeResult)
        assertNotNull(request.evidenceKey)
        assertEquals(RemovalRequestStatus.RESOLVING, request.status)
    }

    @Test fun `evidence stream opens case owned bytes without minting a URL`() {
        val key = "photo-removal/${request.id}/evidence/${UUID.randomUUID()}"
        request.evidenceKey = key
        request.evidenceContentType = "image/png"
        `when`(requests.findById(request.id)).thenReturn(Optional.of(request))
        `when`(storage.open(key)).thenReturn(byteArrayOf(10, 20).inputStream())
        val streamed = service.stream(request.creatorId, request.id)
        assertEquals("image/png", streamed.contentType)
        assertContentEquals(byteArrayOf(10, 20), streamed.stream.use { it.readAllBytes() })
        verify(storage, never()).presignedGetUrl(anyString(), org.mockito.ArgumentMatchers.any() ?: java.time.Duration.ZERO)
    }

    @Test fun `cleanup sweep requests at most fifty due cases`() {
        service.retryEvidenceCleanup()
        val page = org.mockito.ArgumentCaptor.forClass(org.springframework.data.domain.Pageable::class.java)
        verify(requests).findByEvidenceCleanupNextAttemptAtLessThanEqualOrderByEvidenceCleanupNextAttemptAtAscIdAsc(
            org.mockito.ArgumentMatchers.any() ?: java.time.Instant.EPOCH, page.capture() ?: org.springframework.data.domain.PageRequest.of(0, 1))
        assertEquals(50, page.value.pageSize)
    }

    @Test fun `evidence stream identifies the exact case and evidence version read`() {
        request.evidenceKey = "photo-removal/${request.id}/evidence/${UUID.randomUUID()}"
        request.evidenceContentType = "image/png"
        request.version = 7
        request.evidenceGeneration = 3
        `when`(requests.findById(request.id)).thenReturn(Optional.of(request))
        `when`(storage.open(request.evidenceKey!!)).thenReturn(byteArrayOf(10).inputStream())
        val result = service.stream(request.creatorId, request.id)
        result.stream.use {
            assertEquals(7L, result.requestVersion)
            assertEquals(3L, result.evidenceGeneration)
            assertEquals(request.eventIdSnapshot, result.eventId)
            assertEquals(request.evidenceKey, result.evidenceKey)
        }
    }

    private fun image(): SearchImage {
        val output = ByteArrayOutputStream()
        ImageIO.write(BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), "png", output)
        return SearchImage(output.toByteArray(), "image/png", "synthetic.png")
    }
}

class PhotoRemovalIntakeUnitTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val commands = mock(PhotoRemovalCommandRepository::class.java)
    private val events = mock(EventRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val queries = mock(PhotoRemovalQueryService::class.java)
    private val evidence = mock(PhotoRemovalEvidenceService::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val mapper = jacksonObjectMapper().findAndRegisterModules()
    private val service = PhotoRemovalService(requests, commands, events, users, items,
        mock(PhotoRepository::class.java), mock(PhotoRemovalReviewRepository::class.java), mock(PhotoRemovalNotifier::class.java),
        mock(com.quickpitik.service.photos.PhotoAvailabilityService::class.java), evidence, queries,
        mock(PhotoRemovalHistoryRecorder::class.java), PhotoRemovalTransitions(), TransactionTemplate(manager), mapper, Clock.systemUTC(), mock(org.springframework.context.ApplicationEventPublisher::class.java))
    init {
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any() ?: DefaultTransactionDefinition())).thenReturn(SimpleTransactionStatus())
    }

    @Test fun `intake rejects a missing event before writing a case`() {
        val event = UUID.randomUUID()
        `when`(events.findById(event)).thenReturn(Optional.empty())
        val command = com.quickpitik.dto.removal.CreateRemovalRequestCommand(event, "ticket", "contact", null, "privacy", UUID.randomUUID())
        assertEquals(404, assertFailsWith<ApiException> { service.create(UUID.randomUUID(), command) }.status.value())
        assertTrue(mockingDetails(requests).invocations.isEmpty())
    }

    @Test fun `stale cancellation cannot start evidence cleanup`() {
        val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Synthetic run",
            requesterContact = "contact", supportReference = "ticket", requesterReason = "privacy", publicToOwnerReason = "privacy",
            creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(), version = 3)
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
        assertEquals(409, assertFailsWith<ApiException> {
            service.cancel(request.creatorId, request.id, com.quickpitik.dto.removal.CancelRemovalCommand(UUID.randomUUID(), 2, "withdraw"))
        }.status.value())
        assertEquals(RemovalRequestStatus.DRAFT, request.status)
        verifyNoInteractions(evidence)
    }
}

class PhotoRemovalQueryUnitTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val reviews = mock(PhotoRemovalReviewRepository::class.java)
    private val history = mock(PhotoRemovalHistoryRepository::class.java)
    private val service = PhotoRemovalQueryService(requests, items, reviews, history,
        mock(PhotoRepository::class.java), mock(UserRepository::class.java), mock(PhotoRemovalCommandRepository::class.java),
        mock(PhotoRemovalEvidenceService::class.java), mock(com.quickpitik.service.photos.PhotoSearchService::class.java),
        mock(StorageService::class.java), com.quickpitik.config.StorageProperties(), TransactionTemplate(mock(PlatformTransactionManager::class.java)),
        jacksonObjectMapper().findAndRegisterModules(), Clock.systemUTC())
    private val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Synthetic run",
        requesterContact = "private contact", supportReference = "private ticket", requesterReason = "privacy", publicToOwnerReason = "privacy",
        creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID())

    @Test fun `detail counts actual selected states and never exposes evidence keys`() {
        request.evidenceKey = "photo-removal/private-key"
        `when`(requests.findById(request.id)).thenReturn(Optional.of(request))
        `when`(items.findByRequestIdOrderByIdAsc(request.id)).thenReturn(listOf(PhotoRemovalItem(requestId = request.id,
            photoIdSnapshot = UUID.randomUUID(), state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL)))
        val detail = service.adminDetail(request.creatorId, request.id)
        assertEquals(1, detail.phaseCounts.verified)
        assertEquals(0, detail.phaseCounts.candidate)
        assertTrue(detail.evidencePresent)
        assertFalse(jacksonObjectMapper().findAndRegisterModules().writeValueAsString(detail).contains("photo-removal/private-key"))
    }

    @Test fun `history rejects missing cases instead of returning a misleading empty page`() {
        assertEquals(404, assertFailsWith<ApiException> { service.history(request.creatorId, request.id, 0, 50) }.status.value())
    }

    @Test fun `list normalizes offset and caps requested limit`() {
        val filter = org.mockito.ArgumentMatchers.any<org.springframework.data.jpa.domain.Specification<PhotoRemovalRequest>>()
            ?: org.springframework.data.jpa.domain.Specification { _, _, builder -> builder.conjunction() }
        val page = org.mockito.ArgumentMatchers.any<org.springframework.data.domain.Pageable>()
            ?: org.springframework.data.domain.PageRequest.of(0, 1)
        `when`(requests.findAll(filter, page)).thenReturn(org.springframework.data.domain.PageImpl(emptyList()))
        val result = service.list(request.creatorId, null, null, null, -5, 500)
        assertEquals(0, result.offset)
        assertEquals(100, result.limit)
        assertTrue(result.items.isEmpty())
    }
}
