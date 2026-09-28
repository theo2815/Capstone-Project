package com.quickpitik.service.photographer

import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.repository.PhotoMetadataRepository
import com.quickpitik.service.image.PhotoMetadataExtractor
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.*

class PhotoMetadataServiceTest {
    private val repository = Mockito.mock(PhotoMetadataRepository::class.java)
    private val extractor = Mockito.mock(PhotoMetadataExtractor::class.java)
    private val manager = Mockito.mock(PlatformTransactionManager::class.java)
    private val clock = Clock.fixed(Instant.parse("2026-09-28T00:00:00Z"), ZoneOffset.UTC)
    private fun <T> anyArg(): T = Mockito.any()
    private fun service(): PhotoMetadataService {
        Mockito.`when`(manager.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
        return PhotoMetadataService(repository, extractor, TransactionTemplate(manager), clock)
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
    fun `settlement outage does not escape or attempt a second settlement`() {
        Mockito.`when`(repository.claimIfDue(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), Mockito.anyInt())).thenReturn(1)
        Mockito.`when`(extractor.extract(anyArg())).thenReturn(PhotoMetadataFields(cameraModel = "R6"))
        Mockito.`when`(repository.settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg())).thenThrow(IllegalStateException("database down"))
        service().extractIfDue(UUID.randomUUID(), byteArrayOf(1))
        Mockito.verify(repository, Mockito.times(1)).settle(anyArg(), anyArg(), Mockito.anyInt(), anyArg(), anyArg(), anyArg())
    }
}
