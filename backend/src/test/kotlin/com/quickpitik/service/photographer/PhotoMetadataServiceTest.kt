package com.quickpitik.service.photographer

import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.config.PhotoMetadataProperties
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.repository.PhotoMetadataRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.image.PhotoMetadataExtractor
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
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
            PhotoMetadataProperties(recoveryEnabled = recoveryEnabled), TransactionTemplate(manager), clock)
    }

    @Test
    fun `fatal parser error settles FAILED without leaking parser values`() {
        Mockito.`when`(repository.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        Mockito.`when`(extractor.extract(anyArg())).thenThrow(IllegalArgumentException("private tag value"))
        var state: PhotoMetadataState? = null
        var fields: PhotoMetadataFields? = PhotoMetadataFields()
        Mockito.`when`(repository.settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg())).thenAnswer {
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
        Mockito.`when`(repository.settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg())).thenAnswer {
            settled = it.getArgument(3)
            1
        }
        service.extractIfDue(UUID.randomUUID(), byteArrayOf(1))
        assertEquals(PhotoMetadataState.READY, settled)
    }

    @Test
    fun `settlement outage does not escape or attempt a second settlement`() {
        Mockito.`when`(repository.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields(cameraModel = "R6"))
        Mockito.`when`(repository.settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg())).thenThrow(IllegalStateException("database down"))
        service().extractIfDue(UUID.randomUUID(), byteArrayOf(1))
        Mockito.verify(repository, Mockito.times(1)).settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg())
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
