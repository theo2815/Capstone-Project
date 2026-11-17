package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoWorkFence
import com.quickpitik.service.photos.PhotoWorkKind
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import org.springframework.jdbc.core.JdbcTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.*

class EndedIndexingClaimIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var fence: PhotoWorkFence
    @Autowired lateinit var jdbc: JdbcTemplate

    @Test fun `released work finishes its output and returns to backlog with a fresh claim while trash stays held`() {
        val event = events.saveAndFlush(Event(slug = "ended-${UUID.randomUUID()}", name = "Cebu run",
            date = LocalDate.of(2026, 10, 6), location = "Cebu", status = EventStatus.ACTIVE))
        fun photo() = photos.saveAndFlush(Photo(eventId = event.id, s3Key = "synthetic/${UUID.randomUUID()}",
            pricePhp = BigDecimal.ONE, uploadedAt = OffsetDateTime.now().minusHours(1)))
        val live = photo()
        val trashed = photo()
        val old = assertNotNull(fence.claim(live.id, PhotoWorkKind.INDEXING))
        val trashClaim = assertNotNull(fence.claim(trashed.id, PhotoWorkKind.INDEXING))
        jdbc.update("UPDATE photos SET indexing_work_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 hour' WHERE id IN (?, ?)", live.id, trashed.id)
        jdbc.update("UPDATE photo_work_outputs SET provider = 'rekognition', expires_at = CURRENT_TIMESTAMP - INTERVAL '1 hour' WHERE token IN (?, ?)", old.token, trashClaim.token)
        jdbc.update("UPDATE photos SET deleted_at = CURRENT_TIMESTAMP WHERE id = ?", trashed.id)

        assertFalse(fence.releaseEndedIndexing(live.id), "expiry cannot permit a replacement before prior cleanup returns")
        fence.confirmPriorCleanup(old)
        assertNotNull(jdbc.queryForObject("SELECT prior_cleanup_finished_at FROM photo_work_outputs WHERE token = ?",
            OffsetDateTime::class.java, old.token))
        assertTrue(fence.releaseEndedIndexing(live.id))
        val released = photos.findById(live.id).orElseThrow()
        assertNull(released.indexingWorkToken)
        assertNull(released.indexingWorkExpiresAt)
        assertFalse(photos.hasUnfinishedWork(live.id, "INDEXING"))
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM photo_work_outputs WHERE token = ? AND finished_at IS NOT NULL",
            Int::class.java, old.token))
        assertTrue(photos.findIndexingBacklog(listOf(IndexingStatus.PENDING), 5,
            OffsetDateTime.now(), PageRequest.of(0, 10_000)).any { it.id == live.id })
        val next = assertNotNull(fence.claim(live.id, PhotoWorkKind.INDEXING))
        assertNotEquals(old.token, next.token)
        assertFalse(fence.isCurrent(old))
        fence.finish(old)
        assertEquals(next.token, photos.findById(live.id).orElseThrow().indexingWorkToken)

        assertFalse(fence.releaseEndedIndexing(trashed.id))
        assertEquals(trashClaim.token, photos.findById(trashed.id).orElseThrow().indexingWorkToken)
        assertTrue(photos.hasUnfinishedWork(trashed.id, "INDEXING"))
    }
}
