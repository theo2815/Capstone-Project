package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.RemovalState
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

@Transactional
class PhotoPipelineAgeQueryIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var jdbc: JdbcTemplate

    // Both queries scan every photo, so hide the ones other classes left; rolled back with the test.
    @BeforeEach
    fun ignorePhotosFromOtherClasses() {
        jdbc.update("UPDATE photos SET deleted_at = now() WHERE deleted_at IS NULL")
    }

    @Test
    fun `oldest PROCESSING age ignores deleted and removed photos`() {
        assertNull(photos.findOldestProcessingUpload())
        val oldest = fixtures(PhotoStatus.PROCESSING, IndexingStatus.INDEXED)
        assertEquals(oldest.toInstant(), photos.findOldestProcessingUpload()?.toInstant())
    }

    @Test
    fun `oldest PENDING indexing age ignores deleted and removed photos`() {
        assertNull(photos.findOldestPendingIndexingUpload())
        val oldest = fixtures(PhotoStatus.LIVE, IndexingStatus.PENDING)
        assertEquals(oldest.toInstant(), photos.findOldestPendingIndexingUpload()?.toInstant())
    }

    private fun fixtures(status: PhotoStatus, indexing: IndexingStatus): OffsetDateTime {
        val event = events.saveAndFlush(Event(
            slug = "pipeline-${UUID.randomUUID()}", name = "Pipeline ages", date = LocalDate.of(2026, 11, 7),
            location = "Cebu", status = EventStatus.ACTIVE,
        ))
        val oldest = OffsetDateTime.parse("2026-11-07T00:00:00Z")
        fun photo(age: Long, deleted: Boolean = false, deleting: Boolean = false,
                  removal: RemovalState = RemovalState.NONE) = Photo(
            eventId = event.id, s3Key = "test/${UUID.randomUUID()}", pricePhp = BigDecimal.ZERO,
            uploadedAt = oldest.minusSeconds(age), status = status, indexingStatus = indexing,
            deletedAt = if (deleted || deleting) oldest else null,
            deletionStartedAt = if (deleting) oldest else null, removalState = removal,
        )
        photos.saveAllAndFlush(listOf(
            photo(50, deleted = true), photo(40, deleting = true),
            photo(30, removal = RemovalState.REMOVED), photo(20, removal = RemovalState.REVIEW),
            photo(0), photo(-10),
            photo(100).apply { this.status = PhotoStatus.HIDDEN; indexingStatus = IndexingStatus.INDEXED },
        ))
        return oldest
    }
}
