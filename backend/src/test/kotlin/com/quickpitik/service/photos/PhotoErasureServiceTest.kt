package com.quickpitik.service.photos

import com.quickpitik.dto.photos.*
import com.quickpitik.entity.RemovalActorType
import com.quickpitik.exception.ConflictException
import org.junit.jupiter.api.Test
import java.util.UUID
import kotlin.test.assertFailsWith
import kotlin.test.assertEquals
import org.mockito.Mockito
import java.time.Instant
import java.util.Optional
import com.quickpitik.entity.*
import com.quickpitik.repository.PhotoErasureJobRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate

class PhotoErasureServiceTest {
    private inline fun <reified T> mock(): T = org.mockito.Mockito.mock(T::class.java)
    private val jobs = mock<PhotoErasureJobRepository>()
    private val storage = mock<StorageService>()
    private val manager = mock<PlatformTransactionManager>().also {
        Mockito.`when`(it.getTransaction(Mockito.any())).thenReturn(SimpleTransactionStatus())
    }
    private val service = PhotoErasureService(mock(), jobs, mock(), mock(), mock(), mock(), mock(), mock(), mock(),
        storage, mock(), com.quickpitik.config.PhotoRemovalProperties(), com.quickpitik.config.StorageProperties(),
        TransactionTemplate(manager), com.fasterxml.jackson.module.kotlin.jacksonObjectMapper(), java.time.Clock.systemUTC(),
        io.micrometer.core.instrument.simple.SimpleMeterRegistry())

    @Test fun `privacy purpose alone never authorizes resource deletion`() {
        assertFailsWith<ConflictException> {
            service.request(PhotoErasureCommand(UUID.randomUUID(), PhotoErasurePurpose.PRIVACY_REMOVAL,
                RemovalActorType.ADMIN, UUID.randomUUID(), "verified request", commandId = UUID.randomUUID()))
        }
    }

    @Test fun `blank authorization reason is rejected`() {
        assertFailsWith<ConflictException> {
            service.request(PhotoErasureCommand(UUID.randomUUID(), PhotoErasurePurpose.PHOTOGRAPHER_DELETE,
                RemovalActorType.PHOTOGRAPHER, UUID.randomUUID(), " ", commandId = UUID.randomUUID()))
        }
    }

    @Test fun `attempt rejects an unknown job before storage work`() {
        assertFailsWith<com.quickpitik.exception.NotFoundException> { service.attempt(UUID.randomUUID()) }
        Mockito.verifyNoInteractions(storage)
    }

    @Test fun `completed job replay performs no storage work`() {
        val job = job().also { it.outcome = PhotoErasureOutcome.COMPLETE }
        Mockito.`when`(jobs.findById(job.id)).thenReturn(Optional.of(job))
        Mockito.`when`(jobs.lock(job.id)).thenReturn(job)
        assertEquals(PhotoErasureOutcome.COMPLETE, service.attempt(job.id))
        Mockito.verifyNoInteractions(storage)
    }

    @Test fun `unexpired eraser lease remains pending without another storage attempt`() {
        val job = job().also { it.leaseUntil = Instant.now().plusSeconds(3600); it.leaseToken = UUID.randomUUID() }
        Mockito.`when`(jobs.findById(job.id)).thenReturn(Optional.of(job))
        Mockito.`when`(jobs.lock(job.id)).thenReturn(job)
        assertEquals(PhotoErasureOutcome.PENDING, service.attempt(job.id))
        Mockito.verifyNoInteractions(storage)
    }

    @Test fun `late receipt is retained once without rewriting the initial manifest`() {
        val job = job().also { it.outcome = PhotoErasureOutcome.COMPLETE; it.completedAt = Instant.now() }
        val initial = job.manifest
        val receipt = PhotoUploadIntent(photographerId = job.ownerId!!, eventId = job.eventId,
            contentHash = "a".repeat(64), contentType = "image/jpeg", sizeBytes = 1,
            stagingKey = "late/staging", originalKey = "late/original",
            putExpiresAt = java.time.OffsetDateTime.now().plusMinutes(15), expiresAt = java.time.OffsetDateTime.now().plusDays(1))
        Mockito.`when`(jobs.findByPhotoId(job.photoId)).thenReturn(job)
        Mockito.`when`(jobs.lock(job.id)).thenReturn(job)
        service.recordUploadIntent(job.photoId, receipt)
        service.recordUploadIntent(job.photoId, receipt)
        assertEquals(initial, job.manifest)
        assertEquals(1, job.additionalIntents.size)
        assertEquals("late/original", job.additionalIntents.single().originalKey)
        assertEquals(1L, job.revision)
        assertEquals(PhotoErasureOutcome.PENDING, job.outcome)
        kotlin.test.assertNull(job.completedAt)
        Mockito.verifyNoInteractions(storage)
    }

    @Test fun `the erasure manifest always lists the grid tile beside the preview`() {
        val photos = mock<com.quickpitik.repository.PhotoRepository>()
        val eraser = PhotoErasureService(photos, jobs, mock(), mock(), mock(), mock(), mock(), mock(), mock(),
            storage, mock(), com.quickpitik.config.PhotoRemovalProperties(), com.quickpitik.config.StorageProperties(),
            TransactionTemplate(manager), com.fasterxml.jackson.module.kotlin.jacksonObjectMapper(), java.time.Clock.systemUTC(),
            io.micrometer.core.instrument.simple.SimpleMeterRegistry())
        val owner = UUID.randomUUID()
        val photo = Photo(eventId = UUID.randomUUID(), photographerId = owner, s3Key = "original.jpg",
            pricePhp = java.math.BigDecimal("199.00")).also { it.deletedAt = java.time.OffsetDateTime.now() }
        Mockito.`when`(photos.findAllByIdForUpdate(listOf(photo.id))).thenReturn(listOf(photo))
        var saved: PhotoErasureJob? = null
        Mockito.doAnswer { saved = it.getArgument(0); saved }.`when`(jobs).saveAndFlush(Mockito.any(PhotoErasureJob::class.java))

        eraser.request(PhotoErasureCommand(photo.id, PhotoErasurePurpose.PHOTOGRAPHER_DELETE,
            RemovalActorType.PHOTOGRAPHER, owner, "Photographer permanent deletion", commandId = UUID.randomUUID()))

        val keys = saved!!.manifest.keys
        kotlin.test.assertTrue("events/${photo.eventId}/photos/${photo.id}/grid.jpg" in keys, "manifest: $keys")
        kotlin.test.assertTrue("events/${photo.eventId}/photos/${photo.id}/watermark.jpg" in keys, "manifest: $keys")
    }

    @Test fun `makeDue moves a backed-off job to now without other changes`() {
        val job = job().also { it.nextAttemptAt = Instant.now().plusSeconds(3600); it.outcome = PhotoErasureOutcome.RETRY_REQUIRED }
        service.makeDue(job)
        kotlin.test.assertTrue(!job.nextAttemptAt.isAfter(Instant.now()))
        assertEquals(PhotoErasureOutcome.RETRY_REQUIRED, job.outcome)
        Mockito.verify(jobs).saveAndFlush(job)
        Mockito.verifyNoInteractions(storage)
    }

    @Test fun `a refused face cleanup is logged with its reason`() {
        val photos = mock<com.quickpitik.repository.PhotoRepository>()
        val faces = mock<PhotoFaceErasureService>()
        val eraser = PhotoErasureService(photos, jobs, mock(), mock(), mock(), mock(), mock(), mock(), mock(),
            storage, faces, com.quickpitik.config.PhotoRemovalProperties(), com.quickpitik.config.StorageProperties(),
            TransactionTemplate(manager), com.fasterxml.jackson.module.kotlin.jacksonObjectMapper(), java.time.Clock.systemUTC(),
            io.micrometer.core.instrument.simple.SimpleMeterRegistry())
        val job = job(neverIndexed = false)
        Mockito.`when`(jobs.findById(job.id)).thenReturn(Optional.of(job))
        Mockito.`when`(jobs.lock(job.id)).thenReturn(job)
        Mockito.`when`(jobs.saveAndFlush(job)).thenReturn(job) // attempt() uses the saved job as its lease claim
        Mockito.`when`(faces.erase(job.eventId, job.photoId)).thenThrow(com.quickpitik.service.ai.AiApiException(
            org.springframework.http.HttpStatus.SERVICE_UNAVAILABLE, "AI_API_UNAVAILABLE", "Recorded face provider unresolved"))
        val logger = org.slf4j.LoggerFactory.getLogger(PhotoErasureService::class.java) as ch.qos.logback.classic.Logger
        val appender = ch.qos.logback.core.read.ListAppender<ch.qos.logback.classic.spi.ILoggingEvent>().apply { start() }
        logger.addAppender(appender)
        try {
            assertEquals(PhotoErasureOutcome.RETRY_REQUIRED, eraser.attempt(job.id))
            val warn = appender.list.single { it.level == ch.qos.logback.classic.Level.WARN }
            kotlin.test.assertTrue(warn.formattedMessage.contains(job.id.toString()))
            kotlin.test.assertTrue(warn.formattedMessage.contains("Recorded face provider unresolved"))
        } finally {
            logger.detachAppender(appender)
            appender.stop()
        }
    }

    private fun job(neverIndexed: Boolean = true) = PhotoErasureJob(photoId = UUID.randomUUID(), eventId = UUID.randomUUID(),
        ownerId = UUID.randomUUID(), purpose = PhotoErasurePurpose.PHOTOGRAPHER_DELETE,
        authorizedActorType = RemovalActorType.PHOTOGRAPHER, authorizedActorId = UUID.randomUUID(),
        reason = "Photographer permanent deletion", requestId = null, itemId = null, commandId = UUID.randomUUID(),
        manifest = ErasureManifest(listOf("original"), emptyList(), neverIndexed), generation = 0,
        nextAttemptAt = Instant.now(), createdAt = Instant.now())
}
