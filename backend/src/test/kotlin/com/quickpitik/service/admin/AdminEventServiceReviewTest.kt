package com.quickpitik.service.admin

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.aop.framework.ProxyFactory
import org.springframework.transaction.annotation.AnnotationTransactionAttributeSource
import org.springframework.transaction.interceptor.TransactionInterceptor
import org.springframework.transaction.support.TransactionSynchronization
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Admin review of photographer-owned events (V46). One approve + one reject
// endpoint serve both queue states: an initial submission (PENDING, DRAFT)
// and a pricing edit request on a live event (CHANGE_PENDING, ACTIVE).
class AdminEventServiceReviewTest {
    private val adminId = UUID.randomUUID()
    private val photographerId = UUID.randomUUID()
    private lateinit var eventRepository: EventRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var decisions: AdminDecisionLogService
    private lateinit var service: AdminEventService
    private val pushed = mutableListOf<PhotographerMessageKind>()
    private val transactions = testTransactionTemplate()
    private lateinit var covers: EventCoverService
    private lateinit var provider: FaceBibProvider
    private lateinit var aiProperties: AiApiProperties

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        decisions = Mockito.mock(AdminDecisionLogService::class.java)
        Mockito.`when`(eventRepository.save(anyArg())).thenAnswer { it.arguments[0] }
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetEventId = UUID.randomUUID(), decision = "x"))
        Mockito.`when`(decisions.pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())).thenAnswer { inv ->
            pushed += inv.getArgument<PhotographerMessageKind>(1)
            null
        }
        covers = Mockito.mock(EventCoverService::class.java)
        provider = Mockito.mock(FaceBibProvider::class.java)
        aiProperties = Mockito.mock(AiApiProperties::class.java)
        val target = AdminEventService(
            eventRepository,
            photoRepository,
            decisions,
            EventDtoMapper(Mockito.mock(StorageService::class.java), StorageProperties(), photoRepository),
            covers,
            provider,
            aiProperties,
            Mockito.mock(PhotographerSettingsRepository::class.java),
            Mockito.mock(UserRepository::class.java),
            transactions,
            Mockito.mock(AdminActionCommandService::class.java),
            com.quickpitik.service.reference.RegionsService(com.fasterxml.jackson.module.kotlin.jacksonObjectMapper()).apply { load() },
        )
        service = ProxyFactory(target).apply {
            addAdvice(TransactionInterceptor().apply {
                transactionManager = transactions.transactionManager
                transactionAttributeSource = AnnotationTransactionAttributeSource()
            })
        }.proxy as AdminEventService
    }

    @Test
    fun `approving a submission opens the event and tells the photographer`() {
        val event = submitted()

        service.approve(adminId, event.id)

        assertEquals(EventStatus.ACTIVE, event.status)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertEquals(adminId, event.reviewedBy)
        assertNotNull(event.reviewedAt)
        assertEquals(listOf(PhotographerMessageKind.EVENT_APPROVED), pushed)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_approved"), anyArg(), anyArg())
    }

    @Test
    fun `approving a pricing change applies it, re-prices photos and re-renders previews`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "free", "pricePerPhoto" to "0.00", "watermarkPolicy" to "own")
        }

        service.approve(adminId, event.id)

        assertEquals(EventPricingMode.FREE, event.pricingMode)
        assertEquals(0, event.pricePerPhoto.signum())
        assertEquals(WatermarkPolicy.OWN, event.watermarkPolicy)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertEquals(EventStatus.ACTIVE, event.status)
        Mockito.verify(photoRepository).updatePriceByEventId(eqArg(event.id), anyArg())
        Mockito.verify(photoRepository).resetForRewatermark(event.id)
        assertEquals(listOf(PhotographerMessageKind.EVENT_CHANGE_APPROVED), pushed)
    }

    @Test
    fun `approving a submission takes no photo locks`() {
        val event = submitted()

        service.approve(adminId, event.id)

        Mockito.verify(photoRepository, Mockito.never()).lockIdsByEventIdForUpdate(anyArg())
    }

    @Test
    fun `approving a pricing change locks its photos before the event row`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "paid", "pricePerPhoto" to "200.00", "watermarkPolicy" to "platform")
        }

        service.approve(adminId, event.id)

        val order = Mockito.inOrder(photoRepository, eventRepository)
        order.verify(photoRepository).lockIdsByEventIdForUpdate(event.id)
        order.verify(eventRepository).findByIdForReview(event.id)
    }

    @Test
    fun `a pricing change that appears after the unlocked read is refused, not applied without photo locks`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "paid", "pricePerPhoto" to "200.00", "watermarkPolicy" to "platform")
        }
        Mockito.`when`(eventRepository.findReviewStatus(event.id)).thenReturn(EventReviewStatus.PENDING)

        assertFailsWith<ConflictException> { service.approve(adminId, event.id) }

        assertEquals(EventReviewStatus.CHANGE_PENDING, event.reviewStatus)
        Mockito.verify(photoRepository, Mockito.never()).updatePriceByEventId(anyArg(), anyArg())
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `a price-only change does not re-render previews`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "paid", "pricePerPhoto" to "200.00", "watermarkPolicy" to "platform")
        }

        service.approve(adminId, event.id)

        assertEquals(0, BigDecimal("200.00").compareTo(event.pricePerPhoto))
        Mockito.verify(photoRepository).updatePriceByEventId(eqArg(event.id), anyArg())
        Mockito.verify(photoRepository, Mockito.never()).resetForRewatermark(anyArg())
    }

    @Test
    fun `approve and reject refuse an event that is not in the queue`() {
        val event = live()

        assertFailsWith<ConflictException> { service.approve(adminId, event.id) }
        assertFailsWith<ConflictException> { service.reject(adminId, event.id, "no") }
        assertEquals(EventStatus.ACTIVE, event.status)
    }

    @Test
    fun `rejecting a submission keeps it a draft with the reason`() {
        val event = submitted()

        service.reject(adminId, event.id, "Price too high for a fun run")

        assertEquals(EventStatus.DRAFT, event.status)
        assertEquals(EventReviewStatus.REJECTED, event.reviewStatus)
        assertEquals("Price too high for a fun run", event.reviewNote)
        assertEquals(listOf(PhotographerMessageKind.EVENT_REJECTED), pushed)
    }

    @Test
    fun `rejecting a pricing change leaves the live event untouched`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "free", "pricePerPhoto" to "0.00", "watermarkPolicy" to "none")
        }

        service.reject(adminId, event.id, "Keep it paid")

        assertEquals(EventPricingMode.PAID, event.pricingMode)
        assertEquals(0, BigDecimal("150.00").compareTo(event.pricePerPhoto))
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertEquals("Keep it paid", event.reviewNote)
        assertEquals(EventStatus.ACTIVE, event.status)
        assertEquals(listOf(PhotographerMessageKind.EVENT_CHANGE_REJECTED), pushed)
        Mockito.verify(photoRepository, Mockito.never()).resetForRewatermark(anyArg())
    }

    @Test
    fun `approve and reject read the row under a lock`() {
        val event = submitted()

        service.approve(adminId, event.id)

        Mockito.verify(eventRepository).findByIdForReview(event.id)
        Mockito.verify(eventRepository, Mockito.never()).findById(anyArg())
    }

    @Test
    fun `an admin price edit on a free event is refused`() {
        val event = live().apply {
            pricingMode = EventPricingMode.FREE
            pricePerPhoto = BigDecimal.ZERO
            watermarkPolicy = WatermarkPolicy.OWN
        }

        assertFailsWith<ValidationException> {
            service.update(adminId, event.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("100")))
        }
        assertEquals(0, event.pricePerPhoto.signum())
        Mockito.verify(photoRepository, Mockito.never()).updatePriceByEventId(anyArg(), anyArg())
    }

    // Admin ₱0 parity (2026-09-05): a PAID event at ₱0 could be created but
    // never sold, so zero is refused unless the event is FREE.
    @Test
    fun `an admin price edit to zero is refused on a paid event and allowed on a free one`() {
        val paid = live()
        val ex = assertFailsWith<ValidationException> {
            service.update(adminId, paid.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal.ZERO))
        }
        assertEquals("pricePerPhoto", ex.field)
        assertEquals(BigDecimal("150.00"), paid.pricePerPhoto)

        val free = live().apply {
            pricingMode = EventPricingMode.FREE
            pricePerPhoto = BigDecimal.ZERO
            watermarkPolicy = WatermarkPolicy.OWN
        }
        service.update(adminId, free.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal.ZERO))
        assertEquals(0, free.pricePerPhoto.signum())
    }

    @Test
    fun `the review queue lists only pending submissions and change requests`() {
        val queue = listOf(submitted(), live().apply { reviewStatus = EventReviewStatus.CHANGE_PENDING })
        Mockito.`when`(eventRepository.findByReviewStatusInAndDeletedAtIsNullOrderByCreatedAtAsc(anyArg()))
            .thenReturn(queue)

        val page = service.list(stateFilter = null, review = "queue", params = PaginationParams.of(0, 20))

        assertEquals(2, page.total)
        Mockito.verify(eventRepository, Mockito.never()).pageForAdmin(anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `create uploads before opening the event and audit transaction`() {
        var uploadedId: UUID? = null
        Mockito.`when`(covers.upload(anyArg(), anyArg(), anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            Mockito.verify(eventRepository, Mockito.never()).save(anyArg())
            uploadedId = it.getArgument(0)
            "new-cover"
        }
        assertWritesTransactional()

        val result = service.create(adminId, createRequest(), cover())

        assertEquals(uploadedId, result.id)
        Mockito.verify(eventRepository).save(anyArg())
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(result.id), eqArg("event_created"), anyArg(), anyArg())
    }

    @Test
    fun `update uploads outside the transaction and deletes the old cover only after commit`() {
        val event = live().apply { coverS3Key = "old-cover" }
        var committed = false
        var deleted = false
        Mockito.`when`(covers.upload(eqArg(event.id), anyArg(), anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            "new-cover"
        }
        assertWritesTransactional { committed = true }
        Mockito.doAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            assertTrue(committed)
            deleted = true
            null
        }.`when`(covers).delete("old-cover")

        val result = service.update(
            adminId, event.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("200")), cover(), removeCover = true,
        )

        assertTrue(deleted)
        assertEquals("new-cover", event.coverS3Key)
        assertEquals(BigDecimal("200.00"), result.pricePerPhoto)
        assertEquals("set", (result.adminOverrides.single()["before"] as Map<*, *>)["cover"])
        assertEquals("set", (result.adminOverrides.single()["after"] as Map<*, *>)["cover"])
        Mockito.verify(photoRepository).updatePriceByEventId(event.id, BigDecimal("200.00"))
    }

    @Test
    fun `removing a cover waits for commit and records the override`() {
        val event = live().apply { coverS3Key = "old-cover" }
        var committed = false
        assertWritesTransactional { committed = true }
        Mockito.doAnswer {
            assertTrue(committed)
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            null
        }.`when`(covers).delete("old-cover")

        val result = service.update(adminId, event.id, UpdateAdminEventRequest(), removeCover = true)

        assertNull(event.coverS3Key)
        assertEquals("none", (result.adminOverrides.single()["after"] as Map<*, *>)["cover"])
        Mockito.verify(covers).delete("old-cover")
    }

    @Test
    fun `failed cover upload does not persist a new event or audit`() {
        Mockito.`when`(covers.upload(anyArg(), anyArg(), anyArg())).thenThrow(IllegalArgumentException("bad cover"))

        assertFailsWith<IllegalArgumentException> { service.create(adminId, createRequest(), cover()) }

        Mockito.verify(eventRepository, Mockito.never()).save(anyArg())
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `failed update audit preserves the old cover object`() {
        val event = live().apply { coverS3Key = "old-cover" }
        Mockito.`when`(covers.upload(eqArg(event.id), anyArg(), anyArg())).thenReturn("new-cover")
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenThrow(IllegalStateException("audit unavailable"))

        assertFailsWith<IllegalStateException> {
            service.update(adminId, event.id, UpdateAdminEventRequest(), cover())
        }

        Mockito.verify(covers, Mockito.never()).delete("old-cover")
    }

    @Test
    fun `delete commits its audit before best effort provider erasure`() {
        val event = live()
        var committed = false
        var providerWasTransactional: Boolean? = null
        var providerSawCommit = false
        Mockito.`when`(aiProperties.enabled).thenReturn(true)
        assertWritesTransactional { committed = true }
        Mockito.doAnswer {
            providerWasTransactional = TransactionSynchronizationManager.isActualTransactionActive()
            providerSawCommit = committed
            throw IllegalStateException("provider unavailable")
        }.`when`(provider).deleteFacesByEvent(event.id)

        assertTrue(service.delete(adminId, event.id).removed)

        assertEquals(false, providerWasTransactional)
        assertTrue(providerSawCommit)
        assertNotNull(event.deletedAt)
        assertFalse(service.delete(adminId, event.id).removed)
        assertFalse(service.delete(adminId, UUID.randomUUID()).removed)
        Mockito.verify(provider).deleteFacesByEvent(event.id)
    }

    @Test
    fun `failed delete audit never erases provider faces`() {
        val event = live()
        Mockito.`when`(aiProperties.enabled).thenReturn(true)
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenThrow(IllegalStateException("audit unavailable"))

        assertFailsWith<IllegalStateException> { service.delete(adminId, event.id) }

        Mockito.verifyNoInteractions(provider)
    }

    @Test
    fun `invalid updates are rejected before uploading a cover`() {
        val paid = live()
        assertFailsWith<NotFoundException> {
            service.update(adminId, UUID.randomUUID(), UpdateAdminEventRequest(), cover())
        }
        assertFailsWith<ValidationException> {
            service.update(adminId, paid.id, UpdateAdminEventRequest(date = "invalid"), cover())
        }
        assertFailsWith<ValidationException> {
            service.update(adminId, paid.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal.ZERO), cover())
        }
        val free = live().apply { pricingMode = EventPricingMode.FREE }
        assertFailsWith<ValidationException> {
            service.update(adminId, free.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("100")), cover())
        }
        Mockito.verifyNoInteractions(covers)
    }

    @Test
    fun `delete skips provider erasure while AI is disabled`() {
        val event = live()

        assertTrue(service.delete(adminId, event.id).removed)

        Mockito.verifyNoInteractions(provider)
    }

    @Test
    fun `an edit after deletion is not found and cannot reprice photos`() {
        val event = live().apply { deletedAt = java.time.OffsetDateTime.now() }
        assertFailsWith<NotFoundException> {
            service.update(adminId, event.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("200")))
        }
        Mockito.verifyNoInteractions(decisions)
        Mockito.verify(photoRepository, Mockito.never()).updatePriceByEventId(anyArg(), anyArg())
    }

    @Test
    fun `a repeated unchanged text edit has only one override and audit`() {
        val event = live()
        val request = UpdateAdminEventRequest(title = "New name")
        service.update(adminId, event.id, request)
        service.update(adminId, event.id, request)
        assertEquals(1, event.adminOverrides.size)
        Mockito.verify(decisions, Mockito.times(1)).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_updated"), anyArg(), anyArg())
    }

    @Test
    fun `a deletion during cover upload is rechecked before applying any edit`() {
        val event = live().apply { coverS3Key = "old-cover" }
        Mockito.`when`(covers.upload(eqArg(event.id), anyArg(), anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            event.deletedAt = java.time.OffsetDateTime.now()
            "new-cover"
        }
        assertFailsWith<NotFoundException> {
            service.update(adminId, event.id, UpdateAdminEventRequest(title = "After delete"), cover())
        }
        assertEquals("Fun Run", event.name)
        assertEquals("old-cover", event.coverS3Key)
        Mockito.verifyNoInteractions(decisions)
        Mockito.verify(covers, Mockito.never()).delete("old-cover")
    }

    private fun assertWritesTransactional(onCommit: () -> Unit = {}) {
        Mockito.`when`(eventRepository.save(anyArg())).thenAnswer {
            assertTrue(TransactionSynchronizationManager.isActualTransactionActive())
            TransactionSynchronizationManager.registerSynchronization(object : TransactionSynchronization {
                override fun afterCommit() = onCommit()
            })
            it.arguments[0]
        }
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg())).thenAnswer {
            assertTrue(TransactionSynchronizationManager.isActualTransactionActive())
            AdminDecisionLog(adminId = adminId, targetEventId = it.getArgument(1), decision = it.getArgument(2))
        }
    }

    @Test
    fun `invalid location pairs reject legacy and keyed writes before cover or claim`() {
        val event = submitted()
        for ((region, province) in listOf("region-7" to null, null to "cebu", "" to "",
            "region-7" to "metro-manila", "r".repeat(21) to "cebu", "region-7" to "p".repeat(41))) {
            for (key in listOf(null, com.quickpitik.common.IdempotencyKey(UUID.randomUUID().toString()))) {
                val createError = assertFailsWith<ValidationException> {
                    service.create(adminId, createRequest().copy(regionCode = region, provinceCode = province), cover(), key)
                }
                assertEquals("INVALID_REGION", createError.code)
                val updateError = assertFailsWith<ValidationException> {
                    service.update(adminId, event.id, UpdateAdminEventRequest(regionCode = region, provinceCode = province), cover(), key = key)
                }
                assertEquals("INVALID_REGION", updateError.code)
            }
        }
        Mockito.verifyNoInteractions(covers, decisions)
    }

    // ── "Show on /events" (V89) ──────────────────────────────────────────
    @Test
    fun `listing a public photographer event shows it on events and logs the decision`() {
        val event = live()
        val revision = event.reviewRevision

        val row = service.setListing(adminId, event.id, listed = true)

        assertTrue(event.listedOnEvents)
        assertTrue(row.listedOnEvents)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertEquals(revision, event.reviewRevision)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_listed"), anyArg(), anyArg())
        Mockito.verify(eventRepository).save(event)
    }

    @Test
    fun `listing an admin event is refused because it is always listed`() {
        val event = live().apply { createdBy = null }

        val ex = assertFailsWith<ValidationException> { service.setListing(adminId, event.id, listed = true) }

        assertEquals("Admin events are always listed on /events.", ex.message)
        assertFalse(event.listedOnEvents)
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `listing a private event is refused until the photographer makes it public`() {
        val event = live().apply { visibility = EventVisibility.UNLISTED }

        val ex = assertFailsWith<ValidationException> { service.setListing(adminId, event.id, listed = true) }

        assertEquals("This event is Private. The photographer must make it Public first.", ex.message)
        assertFalse(event.listedOnEvents)
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `unlisting always works even on a private event`() {
        val event = live().apply { visibility = EventVisibility.UNLISTED; listedOnEvents = true }

        val row = service.setListing(adminId, event.id, listed = false)

        assertFalse(event.listedOnEvents)
        assertFalse(row.listedOnEvents)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_unlisted"), anyArg(), anyArg())
    }

    @Test
    fun `listing an unknown event is not found`() {
        assertFailsWith<NotFoundException> { service.setListing(adminId, UUID.randomUUID(), listed = true) }
    }

    @Test
    fun `setting the listing to its current value writes no second audit row`() {
        val event = live().apply { listedOnEvents = true }

        val row = service.setListing(adminId, event.id, listed = true)

        assertTrue(row.listedOnEvents)
        Mockito.verifyNoInteractions(decisions)
    }

    private fun createRequest() = CreateAdminEventRequest(
        title = "Fun Run", date = "2026-10-10", location = "Cebu", pricePerPhoto = BigDecimal("150"),
    )

    private fun cover() = AdminEventService.CoverUpload(byteArrayOf(1), "image/jpeg")

    private fun submitted(): Event = event(EventStatus.DRAFT, EventReviewStatus.PENDING)
    private fun live(): Event = event(EventStatus.ACTIVE, EventReviewStatus.APPROVED)

    private fun event(status: EventStatus, review: EventReviewStatus): Event {
        val event = Event(
            slug = "fun-run-${UUID.randomUUID().toString().take(6)}",
            name = "Fun Run",
            date = LocalDate.of(2026, 10, 10),
            location = "Cebu City",
            status = status,
            pricePerPhoto = BigDecimal("150.00"),
            createdBy = photographerId,
            reviewStatus = review,
        )
        Mockito.`when`(eventRepository.findByIdForReview(event.id)).thenReturn(event)
        Mockito.`when`(eventRepository.findReviewStatus(event.id)).thenAnswer { event.reviewStatus }
        Mockito.`when`(eventRepository.findById(event.id)).thenReturn(Optional.of(event))
        return event
    }

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
