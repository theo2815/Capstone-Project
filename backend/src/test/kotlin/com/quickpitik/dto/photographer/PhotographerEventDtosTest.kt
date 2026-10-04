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
            assertEquals(wire, photo.toLibraryDto(0) { null }.status)
        }
    }
}
