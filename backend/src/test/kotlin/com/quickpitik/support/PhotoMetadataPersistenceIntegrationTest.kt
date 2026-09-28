package com.quickpitik.support

import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.image.PhotoMetadataExtractor
import com.quickpitik.service.photographer.PhotoMetadataService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionSynchronizationManager
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.*

class PhotoMetadataPersistenceIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var metadata: PhotoMetadataRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var transactions: TransactionTemplate
    @Autowired private lateinit var service: PhotoMetadataService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @MockitoBean private lateinit var extractor: PhotoMetadataExtractor

    private fun <T> anyArg(): T = Mockito.any()

    private fun fixture(state: PhotoMetadataState = PhotoMetadataState.PENDING): UUID {
        val event = events.save(Event(slug = "metadata-${UUID.randomUUID()}", name = "Run", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        val photo = photos.save(Photo(eventId = event.id, s3Key = "original-${UUID.randomUUID()}", pricePhp = BigDecimal.ONE, status = PhotoStatus.LIVE))
        metadata.save(PhotoMetadata(photoId = photo.id, originalFilename = "camera.jpg", originalSizeBytes = 42, state = state))
        return photo.id
    }

    private fun claim(id: UUID, token: UUID, now: OffsetDateTime = OffsetDateTime.now()) =
        transactions.execute { metadata.claimIfDue(id, token, 1, now, now.plusMinutes(5), 3) }!!

    private fun settle(id: UUID, token: UUID, version: Int = 1, model: String = "R6") =
        transactions.execute { metadata.settle(id, token, version, PhotoMetadataState.READY, PhotoMetadataFields(cameraModel = model), OffsetDateTime.now()) }!!

    @Test
    fun `simultaneous claims yield one worker and one attempt`() {
        val id = fixture()
        val start = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        try {
            val workers = (1..2).map { pool.submit<Int> { start.await(); claim(id, UUID.randomUUID()) } }
            start.countDown()
            assertEquals(listOf(0, 1), workers.map { it.get(10, TimeUnit.SECONDS) }.sorted())
            val row = metadata.findById(id).orElseThrow()
            assertEquals(PhotoMetadataState.PROCESSING, row.state)
            assertEquals(1, row.attempts)
            assertNotNull(row.leaseToken)
            assertNotNull(row.leaseUntil)
        } finally { pool.shutdownNow() }
    }

    @Test
    fun `expired worker cannot replace newer lease or settle another version`() {
        val id = fixture()
        val old = UUID.randomUUID()
        val newer = UUID.randomUUID()
        val now = OffsetDateTime.now()
        assertEquals(1, claim(id, old, now.minusMinutes(6)))
        assertEquals(1, claim(id, newer, now))
        assertEquals(0, settle(id, old))
        assertEquals(0, settle(id, newer, version = 2))
        assertEquals(1, settle(id, newer, model = "Winning camera"))
        val row = metadata.findById(id).orElseThrow()
        assertEquals("Winning camera", row.fields?.cameraModel)
        assertEquals(2, row.attempts)
        assertNull(row.leaseToken)
        assertNull(row.leaseUntil)
    }

    @Test
    fun `purging prevents claims and settlements and physical purge cannot resurrect metadata`() {
        val id = fixture()
        val token = UUID.randomUUID()
        assertEquals(1, claim(id, token))
        jdbc.update("UPDATE photos SET deleted_at = now(), deletion_started_at = now() WHERE id = ?", id)
        assertEquals(0, settle(id, token))
        jdbc.update("DELETE FROM photos WHERE id = ?", id)
        assertEquals(0, settle(id, token))
        assertEquals(0, claim(id, UUID.randomUUID()))
        assertFalse(metadata.existsById(id))
        val purging = fixture()
        jdbc.update("UPDATE photos SET deleted_at = now(), deletion_started_at = now() WHERE id = ?", purging)
        assertEquals(0, claim(purging, UUID.randomUUID()))
    }

    @Test
    fun `terminal current version rows skip parsing and future retries and exhausted leases stay untouched`() {
        for (state in listOf(PhotoMetadataState.READY, PhotoMetadataState.EMPTY, PhotoMetadataState.FAILED, PhotoMetadataState.UNAVAILABLE)) {
            val id = fixture(state)
            service.extractIfDue(id, byteArrayOf(1))
            assertEquals(state, metadata.findById(id).orElseThrow().state)
            assertEquals(0, metadata.findById(id).orElseThrow().attempts)
        }
        val future = fixture()
        jdbc.update("UPDATE photo_metadata SET next_attempt_at = now() + interval '1 hour' WHERE photo_id = ?", future)
        assertEquals(0, claim(future, UUID.randomUUID()))
        val exhausted = fixture(PhotoMetadataState.PROCESSING)
        jdbc.update("UPDATE photo_metadata SET attempts = 3, lease_until = now() - interval '1 minute' WHERE photo_id = ?", exhausted)
        assertEquals(0, claim(exhausted, UUID.randomUUID()))
        Mockito.verifyNoInteractions(extractor)
    }

    @Test
    fun `parse suspends an ambient transaction and persists normalized fields independently`() {
        val id = fixture()
        Mockito.`when`(extractor.extract(anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            PhotoMetadataFields(cameraModel = "R6", iso = 400)
        }
        transactions.executeWithoutResult { status ->
            service.extractIfDue(id, byteArrayOf(1))
            status.setRollbackOnly()
        }
        val row = metadata.findById(id).orElseThrow()
        assertEquals(PhotoMetadataState.READY, row.state)
        assertEquals("R6", row.fields?.cameraModel)
        assertEquals(400L, row.fields?.iso)
        assertEquals("camera.jpg", row.originalFilename)
        assertEquals(42L, row.originalSizeBytes)
        assertNotNull(row.extractedAt)
        assertNull(row.nextAttemptAt)
        assertEquals(PhotoStatus.LIVE, photos.findById(id).orElseThrow().status)
    }

    @Test
    fun `empty tags settle EMPTY and fatal parser failure is terminal without changing LIVE`() {
        val empty = fixture()
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields())
        service.extractIfDue(empty, byteArrayOf(1))
        assertEquals(PhotoMetadataState.EMPTY, metadata.findById(empty).orElseThrow().state)
        val failed = fixture()
        Mockito.`when`(extractor.extract(anyArg())).thenThrow(IllegalArgumentException("corrupt tags"))
        service.extractIfDue(failed, byteArrayOf(1))
        service.extractIfDue(failed, byteArrayOf(1))
        val row = metadata.findById(failed).orElseThrow()
        assertEquals(PhotoMetadataState.FAILED, row.state)
        assertEquals(1, row.attempts)
        assertNull(row.nextAttemptAt)
        assertNull(row.leaseToken)
        assertNull(row.fields)
        assertEquals(PhotoStatus.LIVE, photos.findById(failed).orElseThrow().status)
    }
}
