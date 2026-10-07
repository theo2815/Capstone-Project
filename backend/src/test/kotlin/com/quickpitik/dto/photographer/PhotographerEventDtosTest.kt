package com.quickpitik.dto.photographer

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import org.junit.jupiter.api.Test
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

class PhotographerEventDtosTest {
    @Test
    fun `mapper signs only the supplied fresh availability and preserves blocked photo identity`() {
        val storage = org.mockito.Mockito.mock(com.quickpitik.service.storage.StorageService::class.java)
        val settings = com.quickpitik.config.StorageProperties()
        val mapper = com.quickpitik.service.photographer.PhotographerPhotoDtoMapper(storage, settings)
        val photo = Photo(eventId = UUID.randomUUID(), s3Key = "original.jpg", thumbnailS3Key = "thumbnail.jpg", pricePhp = BigDecimal.TEN)
        val blocked = mapper.toLibraryDto(photo, 2, false)
        assertEquals(photo.id, blocked.id)
        assertEquals(2, blocked.salesCount)
        assertEquals(false, blocked.mediaAvailable)
        assertEquals(null, blocked.thumbnailUrl)
        org.mockito.Mockito.verifyNoInteractions(storage)
        org.mockito.Mockito.`when`(storage.presignedGetUrl("thumbnail.jpg", settings.presignedTtl.thumbnail)).thenReturn("signed-thumbnail")
        val allowed = mapper.toLibraryDto(photo, 2, true)
        assertEquals(true, allowed.mediaAvailable)
        assertEquals("signed-thumbnail", allowed.thumbnailUrl)
        org.mockito.Mockito.verify(storage).presignedGetUrl("thumbnail.jpg", settings.presignedTtl.thumbnail)
    }

    @Test
    fun `detail carries removal time for private history`() {
        val photographerId = UUID.randomUUID()
        val removedAt = OffsetDateTime.parse("2026-09-04T21:36:27Z")
        val event = Event(
            slug = "removed-${UUID.randomUUID()}",
            name = "Removed event",
            date = LocalDate.of(2026, 9, 1),
            location = "Cebu",
            status = EventStatus.ARCHIVED,
            deletedAt = removedAt,
            createdBy = photographerId,
        )
        val detail = detailDto(event, EventPhotographer(EventPhotographerId(event.id, photographerId)), null)

        assertEquals(removedAt, detail.deletedAt)
    }

    @Test
    fun `library preserves processing live and hidden statuses`() {
        val photo = Photo(eventId = UUID.randomUUID(), s3Key = "original.jpg", pricePhp = BigDecimal.TEN)
        for ((status, wire) in listOf(PhotoStatus.PROCESSING to "processing", PhotoStatus.LIVE to "live", PhotoStatus.HIDDEN to "hidden")) {
            photo.status = status
            assertEquals(wire, photo.toLibraryDto(0, true) { null }.status)
        }
    }
}
