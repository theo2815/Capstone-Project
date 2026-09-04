package com.quickpitik.service.events

import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

// Public event reads after V46: a pending/rejected (DRAFT) event is not a
// public page, an UNLISTED live event is reachable by its link, and the
// detail names its owner so a free gallery can say whose it is.
class EventServiceTest {
    private val eventRepository = Mockito.mock(EventRepository::class.java)
    private val settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
    private val mapper = EventDtoMapper(Mockito.mock(StorageService::class.java), StorageProperties())
    private val service = EventService(eventRepository, mapper, settingsRepository)

    @Test
    fun `detail carries visibility, pricing and the owner's handle`() {
        val owner = UUID.randomUUID()
        val event = event(EventStatus.ACTIVE).apply {
            createdBy = owner
            visibility = EventVisibility.UNLISTED
            pricingMode = EventPricingMode.FREE
            watermarkPolicy = WatermarkPolicy.NONE
        }
        Mockito.`when`(eventRepository.findPublicBySlug(event.slug)).thenReturn(event)
        Mockito.`when`(settingsRepository.findById(owner))
            .thenReturn(Optional.of(PhotographerSettings(userId = owner, handle = "paksitphotos")))

        val dto = service.findBySlug(event.slug)!!

        assertEquals("unlisted", dto.visibility)
        assertEquals("free", dto.pricingMode)
        assertEquals("paksitphotos", dto.photographerHandle)
    }

    @Test
    fun `an admin event has no owner handle and the pre-V46 defaults`() {
        val event = event(EventStatus.ACTIVE)
        Mockito.`when`(eventRepository.findPublicBySlug(event.slug)).thenReturn(event)

        val dto = service.findBySlug(event.slug)!!

        assertEquals("public", dto.visibility)
        assertEquals("paid", dto.pricingMode)
        assertNull(dto.photographerHandle)
        Mockito.verifyNoInteractions(settingsRepository)
    }

    @Test
    fun `a slug the public lookup does not return is not found`() {
        Mockito.`when`(eventRepository.findPublicBySlug("pending-run")).thenReturn(null)

        assertNull(service.findBySlug("pending-run"))
    }

    private fun event(status: EventStatus) = Event(
        slug = "run-${UUID.randomUUID().toString().take(6)}",
        name = "Run",
        date = LocalDate.of(2026, 10, 10),
        location = "Cebu City",
        status = status,
    )
}
