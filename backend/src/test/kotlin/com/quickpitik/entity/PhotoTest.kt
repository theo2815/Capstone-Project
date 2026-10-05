package com.quickpitik.entity

import org.junit.jupiter.api.Test
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PhotoTest {

    @Test
    fun `removal review and authorized removal are not sellable`() {
        for (state in listOf(RemovalState.REVIEW, RemovalState.REMOVED)) {
            val reviewPhoto = photo(PhotoStatus.LIVE).apply { removalState = state }
            assertFalse(reviewPhoto.isSellable)
        }
    }

    private fun photo(status: PhotoStatus, deletedAt: OffsetDateTime? = null) = Photo(
        eventId = UUID.randomUUID(),
        s3Key = "originals/photo.jpg",
        pricePhp = BigDecimal("125.00"),
        status = status,
        deletedAt = deletedAt,
    )

    @Test
    fun `a live photo that is not trashed is sellable`() {
        assertTrue(photo(PhotoStatus.LIVE).isSellable)
    }

    @Test
    fun `no other status is sellable`() {
        PhotoStatus.entries.filter { it != PhotoStatus.LIVE }.forEach {
            assertFalse(photo(it).isSellable, "$it must not be sellable")
        }
    }

    @Test
    fun `a trashed photo is not sellable even while LIVE`() {
        assertFalse(photo(PhotoStatus.LIVE, deletedAt = OffsetDateTime.now()).isSellable)
    }
}
