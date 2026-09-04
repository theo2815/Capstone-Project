package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.CreateMyEventRequest
import com.quickpitik.dto.photographer.UpdateMyEventRequest
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.websocket.AdminInboxEvent
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Photographer-owned events (V46). Every one is reviewed before it goes live,
// and once live the pricing trio (paid/free · price · watermark) can only
// change through an admin-approved request — the backend, not the UI, is what
// stops a photographer from flipping a live event free.
class PhotographerOwnedEventServiceTest {
    private val photographerId = UUID.randomUUID()
    private lateinit var eventRepository: EventRepository
    private lateinit var coverageRepository: EventPhotographerRepository
    private lateinit var settingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var eventPublisher: ApplicationEventPublisher
    private lateinit var service: PhotographerOwnedEventService
    private val saved = mutableListOf<Event>()
    private val published = mutableListOf<Any>()

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        coverageRepository = Mockito.mock(EventPhotographerRepository::class.java)
        settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
        val mapper = Mockito.mock(EventDtoMapper::class.java)
        Mockito.`when`(eventRepository.save(anyArg())).thenAnswer { (it.arguments[0] as Event).also { e -> saved += e } }
        Mockito.`when`(coverageRepository.save(anyArg())).thenAnswer { it.arguments[0] }
        Mockito.doAnswer { published += it.arguments[0]; null }.`when`(eventPublisher).publishEvent(anyArg<Any>())
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(
            Optional.of(User(id = photographerId, email = "p@test.local", passwordHash = "x", name = "Photog", role = Role.PHOTOGRAPHER)),
        )
        approved()
        service = PhotographerOwnedEventService(
            eventRepository,
            coverageRepository,
            settingsRepository,
            userRepository,
            Mockito.mock(EventCoverService::class.java),
            mapper,
            eventPublisher,
        )
    }

    @Test
    fun `only an approved photographer may create an event`() {
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(
            Optional.of(PhotographerSettings(userId = photographerId, verificationStatus = VerificationStatus.PENDING)),
        )
        val ex = assertFailsWith<ApiException> { service.create(photographerId, paidRequest(), null) }
        assertEquals(ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED, ex.code)
    }

    @Test
    fun `a new paid event starts as a draft awaiting review with coverage and an admin inbox ping`() {
        val dto = service.create(photographerId, paidRequest(visibility = "unlisted"), null)

        val event = saved.single()
        assertEquals(EventStatus.DRAFT, event.status)
        assertEquals(EventReviewStatus.PENDING, event.reviewStatus)
        assertEquals(photographerId, event.createdBy)
        assertEquals(EventVisibility.UNLISTED, event.visibility)
        assertEquals(EventPricingMode.PAID, event.pricingMode)
        assertEquals(WatermarkPolicy.PLATFORM, event.watermarkPolicy)
        assertEquals(0, BigDecimal("150.00").compareTo(event.pricePerPhoto))
        assertTrue(event.slug.startsWith("cebu-fun-run-"))
        Mockito.verify(coverageRepository).save(anyArg<EventPhotographer>())
        val ping = published.filterIsInstance<AdminInboxEvent>().single()
        assertEquals("event_submitted", ping.payload["type"])
        assertEquals("pending", dto.reviewStatus)
        assertTrue(dto.ownedByMe)
    }

    @Test
    fun `a free event has no price and never the platform mark`() {
        service.create(photographerId, CreateMyEventRequest(title = "Free Run", date = "2026-10-10", location = "Talisay, Cebu", pricingMode = "free", watermarkPolicy = "none"), null)

        val event = saved.single()
        assertEquals(EventPricingMode.FREE, event.pricingMode)
        assertEquals(0, event.pricePerPhoto.signum())
        assertEquals(WatermarkPolicy.NONE, event.watermarkPolicy)

        val ex = assertFailsWith<ValidationException> {
            service.create(photographerId, CreateMyEventRequest(title = "Free Run", date = "2026-10-10", location = "Cebu", pricingMode = "free", watermarkPolicy = "platform"), null)
        }
        assertEquals("watermarkPolicy", ex.field)
    }

    @Test
    fun `a paid event needs a positive price`() {
        val ex = assertFailsWith<ValidationException> {
            service.create(photographerId, CreateMyEventRequest(title = "Run", date = "2026-10-10", location = "Cebu", pricingMode = "paid"), null)
        }
        assertEquals("pricePerPhoto", ex.field)
    }

    @Test
    fun `another photographer's event is invisible to update`() {
        Mockito.`when`(eventRepository.findByIdAndCreatedByAndDeletedAtIsNull(anyArg(), anyArg())).thenReturn(null)

        assertFailsWith<NotFoundException> {
            service.update(photographerId, UUID.randomUUID(), UpdateMyEventRequest(title = "Mine now"), null)
        }
    }

    @Test
    fun `a pricing edit on a live event does not touch the event and lands in the queue`() {
        val event = liveEvent()

        val dto = service.update(photographerId, event.id, UpdateMyEventRequest(pricingMode = "free", watermarkPolicy = "own"), null)

        assertEquals(EventPricingMode.PAID, event.pricingMode)
        assertEquals(0, BigDecimal("150.00").compareTo(event.pricePerPhoto))
        assertEquals(WatermarkPolicy.PLATFORM, event.watermarkPolicy)
        assertEquals(EventStatus.ACTIVE, event.status)
        assertEquals(EventReviewStatus.CHANGE_PENDING, event.reviewStatus)
        assertEquals("free", event.pendingChange?.get("pricingMode"))
        assertEquals("own", event.pendingChange?.get("watermarkPolicy"))
        assertEquals("event_change_requested", published.filterIsInstance<AdminInboxEvent>().single().payload["type"])
        assertEquals("change_pending", dto.reviewStatus)
    }

    @Test
    fun `withdrawing a pending change returns the live event to approved`() {
        val event = liveEvent().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "free")
        }

        service.update(photographerId, event.id, UpdateMyEventRequest(withdrawPendingChange = true), null)

        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
    }

    @Test
    fun `name and visibility edits on a live event apply immediately`() {
        val event = liveEvent()

        service.update(photographerId, event.id, UpdateMyEventRequest(title = "Renamed Run", visibility = "unlisted"), null)

        assertEquals("Renamed Run", event.name)
        assertEquals(EventVisibility.UNLISTED, event.visibility)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertTrue(published.isEmpty())
    }

    @Test
    fun `editing a rejected event applies the pricing directly and resubmits it`() {
        val event = liveEvent().apply {
            status = EventStatus.DRAFT
            reviewStatus = EventReviewStatus.REJECTED
            reviewNote = "Price too high"
        }

        service.update(photographerId, event.id, UpdateMyEventRequest(pricePerPhoto = BigDecimal("99.00")), null)

        assertEquals(0, BigDecimal("99.00").compareTo(event.pricePerPhoto))
        assertEquals(EventReviewStatus.PENDING, event.reviewStatus)
        assertNull(event.reviewNote)
        assertEquals("event_submitted", published.filterIsInstance<AdminInboxEvent>().single().payload["type"])
        assertNotNull(event.createdBy)
    }

    private fun approved() {
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(
            Optional.of(PhotographerSettings(userId = photographerId, handle = "photog", watermarkS3Key = "w.png", verificationStatus = VerificationStatus.APPROVED)),
        )
    }

    private fun paidRequest(visibility: String = "public") = CreateMyEventRequest(
        title = "Cebu Fun Run",
        date = "2026-10-10",
        location = "IT Park, Cebu City",
        pricingMode = "paid",
        pricePerPhoto = BigDecimal("150"),
        visibility = visibility,
    )

    private fun liveEvent(): Event {
        val event = Event(
            slug = "cebu-fun-run-abc123",
            name = "Cebu Fun Run",
            date = LocalDate.of(2026, 10, 10),
            location = "IT Park, Cebu City",
            status = EventStatus.ACTIVE,
            pricePerPhoto = BigDecimal("150.00"),
            createdBy = photographerId,
            reviewStatus = EventReviewStatus.APPROVED,
        )
        Mockito.`when`(eventRepository.findByIdAndCreatedByAndDeletedAtIsNull(event.id, photographerId)).thenReturn(event)
        Mockito.`when`(coverageRepository.findById(anyArg())).thenReturn(Optional.empty())
        return event
    }

    private fun <T> anyArg(): T = Mockito.any()
}
