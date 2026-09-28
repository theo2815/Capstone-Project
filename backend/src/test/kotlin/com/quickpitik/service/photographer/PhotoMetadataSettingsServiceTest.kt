package com.quickpitik.service.photographer

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PhotoMetadataSettingsServiceTest {
    private val userId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()
    private val key = EventPhotographerId(eventId, userId)
    private val settings = Mockito.mock(PhotographerSettingsRepository::class.java)
    private val photographerSettings = Mockito.mock(PhotographerSettingsService::class.java)
    private val coverage = Mockito.mock(EventPhotographerRepository::class.java)
    private val events = Mockito.mock(EventRepository::class.java)
    private val service = PhotoMetadataSettingsService(settings, photographerSettings, coverage, events)

    @Test
    fun `missing settings have a private default`() {
        Mockito.`when`(settings.findById(userId)).thenReturn(Optional.empty())
        assertEquals(false, service.getDefault(userId).metadataPublicByDefault)
    }

    @Test
    fun `setting the default changes only the photographer settings row`() {
        val row = PhotographerSettings(userId = userId)
        Mockito.`when`(photographerSettings.getOrCreate(userId)).thenReturn(row)
        assertEquals(true, service.setDefault(userId, true).metadataPublicByDefault)
        assertEquals(true, row.metadataPublicByDefault)
        Mockito.verifyNoInteractions(coverage)
    }

    @Test
    fun `event read returns saved choice and a private fallback for a legacy owned event`() {
        Mockito.`when`(coverage.findById(key)).thenReturn(Optional.of(EventPhotographer(key, metadataPublic = true)))
        assertEquals(true, service.getEvent(userId, eventId).metadataPublic)

        Mockito.`when`(coverage.findById(key)).thenReturn(Optional.empty())
        Mockito.`when`(events.findByIdAndCreatedByAndDeletedAtIsNull(eventId, userId)).thenReturn(ownedEvent())
        assertEquals(false, service.getEvent(userId, eventId).metadataPublic)
        Mockito.verify(coverage, Mockito.never()).upsertMetadataVisibility(eventId, userId, false)
    }

    @Test
    fun `event write changes covered events but refuses an unrelated event`() {
        Mockito.`when`(coverage.updateMetadataVisibility(eventId, userId, true)).thenReturn(1)
        assertEquals(true, service.setEvent(userId, eventId, true).metadataPublic)
        Mockito.verify(coverage).updateMetadataVisibility(eventId, userId, true)
        Mockito.verify(coverage, Mockito.never()).upsertMetadataVisibility(eventId, userId, true)

        assertFailsWith<NotFoundException> { service.setEvent(userId, eventId, false) }
        Mockito.verify(coverage, Mockito.never()).upsertMetadataVisibility(eventId, userId, false)
    }

    private fun ownedEvent() = Event(
        id = eventId,
        slug = "owned-$eventId",
        name = "Owned",
        date = LocalDate.now(),
        location = "Cebu",
        status = EventStatus.ACTIVE,
        createdBy = userId,
    )
}
