package com.quickpitik.dto.photographer

import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import org.junit.jupiter.api.Test
import java.math.BigDecimal
import java.util.UUID
import kotlin.test.assertEquals

class PhotographerEventDtosTest {
    @Test
    fun `library preserves processing live and hidden statuses`() {
        val photo = Photo(eventId = UUID.randomUUID(), s3Key = "original.jpg", pricePhp = BigDecimal.TEN)
        for ((status, wire) in listOf(PhotoStatus.PROCESSING to "processing", PhotoStatus.LIVE to "live", PhotoStatus.HIDDEN to "hidden")) {
            photo.status = status
            assertEquals(wire, photo.toLibraryDto(0) { null }.status)
        }
    }
}
