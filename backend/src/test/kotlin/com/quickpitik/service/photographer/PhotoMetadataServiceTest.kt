package com.quickpitik.service.photographer

import com.quickpitik.config.PhotoMetadataProperties
import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.repository.PhotoMetadataRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.image.PhotoMetadataExtractor
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import kotlin.test.*

class PhotoMetadataServiceTest {
    private val repository = Mockito.mock(PhotoMetadataRepository::class.java)
    private val photos = Mockito.mock(PhotoRepository::class.java)
    private val coverage = Mockito.mock(EventPhotographerRepository::class.java)
    private val extractor = Mockito.mock(PhotoMetadataExtractor::class.java)
    private val storage = Mockito.mock(StorageService::class.java)
    private val watermarkPool = Mockito.mock(ThreadPoolTaskExecutor::class.java)
    private val manager = Mockito.mock(PlatformTransactionManager::class.java)
    private val clock = Clock.fixed(Instant.parse("2026-09-28T00:00:00Z"), ZoneOffset.UTC)
    private fun <T> anyArg(): T = Mockito.any()
    private fun service(recoveryEnabled: Boolean = true): PhotoMetadataService {
        Mockito.`when`(manager.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
        return PhotoMetadataService(repository, photos, coverage, extractor, storage, watermarkPool,
            PhotoMetadataProperties(recoveryEnabled = recoveryEnabled), TransactionTemplate(manager), clock, watermarkPool)
    }

    @Test
    fun `dispatch defers claims and oversized or rejected originals remain recoverable`() {
        val service = service()
        val pending = mutableListOf<Runnable>()
        Mockito.doAnswer { pending += it.getArgument<Runnable>(0); null }.`when`(watermarkPool).execute(Mockito.any(Runnable::class.java))
        service.dispatchExtraction(UUID.randomUUID(), byteArrayOf(1))
        assertEquals(1, pending.size)
        Mockito.verifyNoInteractions(repository, extractor, storage)

        service.dispatchExtraction(UUID.randomUUID(), ByteArray(25 * 1024 * 1024 + 1))
        assertEquals(1, pending.size)
        Mockito.doThrow(java.util.concurrent.RejectedExecutionException()).`when`(watermarkPool).execute(Mockito.any(Runnable::class.java))
        service.dispatchExtraction(UUID.randomUUID(), byteArrayOf(2))
        Mockito.verifyNoInteractions(repository, extractor, storage)
        pending.single().run()
        Mockito.verify(repository).claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())
    }

    @Test
    fun `fatal parser error settles FAILED without leaking parser values`() {
        Mockito.`when`(repository.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        Mockito.`when`(extractor.extract(anyArg())).thenThrow(IllegalArgumentException("private tag value"))
        var state: PhotoMetadataState? = null
        var fields: PhotoMetadataFields? = PhotoMetadataFields()
        Mockito.`when`(repository.settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg(), Mockito.isNull<Long>())).thenAnswer {
            state = it.getArgument(3)
            fields = it.getArgument(4)
            1
        }
        service().extractIfDue(UUID.randomUUID(), byteArrayOf(1))
        assertEquals(PhotoMetadataState.FAILED, state)
        assertNull(fields)
    }

    @Test
    fun `claim failure or lost claim never invokes parser`() {
        val service = service()
        service.extractIfDue(UUID.randomUUID(), byteArrayOf(1))
        Mockito.`when`(repository.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenThrow(IllegalStateException("database down"))
        service.extractIfDue(UUID.randomUUID(), byteArrayOf(1))
        Mockito.verifyNoInteractions(extractor)
    }

    @Test
    fun `disabled recovery leaves hot path extraction active`() {
        val service = service(recoveryEnabled = false)
        service.recoverDue()
        Mockito.verifyNoInteractions(storage, watermarkPool)
        Mockito.verify(repository, Mockito.never()).findDueCandidateIds(Mockito.anyInt(), Mockito.anyInt(), anyArg(), anyArg(), anyArg())

        Mockito.`when`(repository.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields(cameraModel = "R6"))
        var settled: PhotoMetadataState? = null
        Mockito.`when`(repository.settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg(), Mockito.isNull<Long>())).thenAnswer {
            settled = it.getArgument(3)
            1
        }
        service.extractIfDue(UUID.randomUUID(), byteArrayOf(1))
        assertEquals(PhotoMetadataState.READY, settled)
    }

    @Test
    fun `recovery submits no more than ten ids with larger idle pool`() {
        val pool = Mockito.mock(ThreadPoolExecutor::class.java)
        Mockito.`when`(watermarkPool.threadPoolExecutor).thenReturn(pool)
        Mockito.`when`(pool.queue).thenReturn(LinkedBlockingQueue<Runnable>())
        Mockito.`when`(watermarkPool.maxPoolSize).thenReturn(12)
        val ids = List(12) { UUID.randomUUID() }
        Mockito.`when`(repository.findDueCandidateIds(Mockito.anyInt(), Mockito.anyInt(), anyArg(), anyArg(), anyArg())).thenAnswer {
            ids.take(it.getArgument<org.springframework.data.domain.Pageable>(4).pageSize)
        }
        val submitted = mutableListOf<Runnable>()
        Mockito.doAnswer { submitted += it.getArgument<Runnable>(0); null }.`when`(watermarkPool).execute(Mockito.any(Runnable::class.java))

        service().recoverDue()

        assertEquals(10, submitted.size)
        Mockito.verifyNoInteractions(storage)
    }

    @Test
    fun `settlement outage does not escape or attempt a second settlement`() {
        Mockito.`when`(repository.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields(cameraModel = "R6"))
        Mockito.`when`(repository.settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg(), Mockito.isNull<Long>())).thenThrow(IllegalStateException("database down"))
        service().extractIfDue(UUID.randomUUID(), byteArrayOf(1))
        Mockito.verify(repository, Mockito.times(1)).settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg(), Mockito.isNull<Long>())
    }

    @Test
    fun `gallery access batches public policy and gives the real owner precedence`() {
        val eventId = UUID.randomUUID()
        val ownerId = UUID.randomUUID()
        val publicId = UUID.randomUUID()
        val privateId = UUID.randomUUID()
        val gallery = List(100) { index ->
            Photo(
                eventId = eventId,
                photographerId = listOf(ownerId, publicId, privateId)[index % 3],
                s3Key = "photo-$index.jpg",
                pricePhp = BigDecimal.ONE,
            )
        }
        Mockito.`when`(coverage.findPublicMetadataPhotographerIds(anyArg(), anyArg()))
            .thenReturn(listOf(ownerId, publicId))

        val access = service().metadataAccess(eventId, gallery, ownerId)

        assertEquals("owner", access.getValue(gallery[0].id))
        assertEquals("public", access.getValue(gallery[1].id))
        assertEquals("none", access.getValue(gallery[2].id))
        Mockito.verify(coverage, Mockito.times(1)).findPublicMetadataPhotographerIds(anyArg(), anyArg())
    }
}
