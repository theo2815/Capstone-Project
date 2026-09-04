package com.quickpitik.service.admin

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

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

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        decisions = Mockito.mock(AdminDecisionLogService::class.java)
        Mockito.`when`(eventRepository.save(anyArg())).thenAnswer { it.arguments[0] }
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetEventId = UUID.randomUUID(), decision = "x"))
        Mockito.`when`(decisions.pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())).thenAnswer { inv ->
            pushed += inv.getArgument<PhotographerMessageKind>(1)
            null
        }
        service = AdminEventService(
            eventRepository,
            photoRepository,
            decisions,
            Mockito.mock(EventDtoMapper::class.java),
            Mockito.mock(EventCoverService::class.java),
            Mockito.mock(FaceBibProvider::class.java),
            AiApiProperties(enabled = false),
            Mockito.mock(PhotographerSettingsRepository::class.java),
            Mockito.mock(UserRepository::class.java),
        )
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

    @Test
    fun `the review queue lists only pending submissions and change requests`() {
        val queue = listOf(submitted(), live().apply { reviewStatus = EventReviewStatus.CHANGE_PENDING })
        Mockito.`when`(eventRepository.findByReviewStatusInAndDeletedAtIsNullOrderByCreatedAtAsc(anyArg()))
            .thenReturn(queue)

        val page = service.list(stateFilter = null, review = "queue", params = PaginationParams.of(0, 20))

        assertEquals(2, page.total)
        Mockito.verify(eventRepository, Mockito.never()).pageForAdmin(anyArg(), anyArg(), anyArg(), anyArg())
    }

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
        Mockito.`when`(eventRepository.findById(event.id)).thenReturn(Optional.of(event))
        return event
    }

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
