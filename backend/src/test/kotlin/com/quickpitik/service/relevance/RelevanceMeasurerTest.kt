package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.PhotoToMeasure
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.ImageCheck
import com.quickpitik.service.ai.LabelHit
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.testTransactionTemplate
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.Executor
import kotlin.test.assertEquals

class RelevanceMeasurerTest {
    private val batches = Mockito.mock(BatchRelevanceRepository::class.java)
    private val storage = Mockito.mock(StorageService::class.java)
    private val client = Mockito.mock(RekognitionAiClient::class.java)
    @Suppress("UNCHECKED_CAST")
    private val provider = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
        .also { Mockito.`when`(it.getIfAvailable()).thenReturn(client) }
    private val registry = SimpleMeterRegistry()
    private val clock = Clock.fixed(Instant.parse("2003-03-04T05:06:07Z"), ZoneOffset.UTC)
    private val now = OffsetDateTime.ofInstant(clock.instant(), ZoneOffset.UTC)
    private val batchId = UUID.randomUUID()
    private val token = UUID.randomUUID()

    // Runs each photo on the calling thread, so the asserts see what the pool thread would.
    private fun measurer() = RelevanceMeasurer(batches, storage, provider, Executor { it.run() }, RelevanceProperties(), clock,
        testTransactionTemplate(), registry)

    private fun photo(): PhotoToMeasure = object : PhotoToMeasure {
        override val photoId: UUID = UUID.randomUUID()
        override val s3Key = "events/e/photos/$photoId/original.jpg"
    }

    private fun bytesOf(p: PhotoToMeasure) = p.photoId.toString().toByteArray()

    private fun counted(result: String) = registry.find("qp.relevance.measured").tag("result", result).counter()?.count() ?: 0.0

    @Test
    fun `each photo is read from storage and checked once`() {
        val labelled = photo()
        val bare = photo()
        listOf(labelled, bare).forEach { Mockito.`when`(storage.getBytes(it.s3Key)).thenReturn(bytesOf(it)) }
        Mockito.`when`(client.imageCheck(bytesOf(labelled))).thenReturn(ImageCheck(50.0, 60.0, 40.0, 30.0,
            listOf(LabelHit("Person", 0.9), LabelHit("Marathon", 0.7), LabelHit("Running", 0.8), LabelHit("Tree", 0.99))))
        Mockito.`when`(client.imageCheck(bytesOf(bare))).thenReturn(ImageCheck(null, null, 10.0, 2.0, emptyList()))
        Mockito.`when`(batches.saveMeasurement(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(),
            Mockito.anyDouble(), Mockito.anyDouble())).thenReturn(1)

        assertEquals(2, measurer().measure(batchId, token, listOf(labelled, bare)))

        Mockito.verify(storage).getBytes(labelled.s3Key)
        Mockito.verify(storage).getBytes(bare.s3Key)
        Mockito.verify(client, Mockito.times(2)).imageCheck(anyArg())
        // The best race label is kept; an absent label is stored as 0, never null.
        Mockito.verify(batches).saveMeasurement(labelled.photoId, batchId, token, now, 50.0, 60.0, 40.0, 30.0, 0.9, 0.8)
        Mockito.verify(batches).saveMeasurement(bare.photoId, batchId, token, now, null, null, 10.0, 2.0, 0.0, 0.0)
        assertEquals(2.0, counted("ok"))
        assertEquals(2L, registry.timer("qp.relevance.measure").count())
    }

    @Test
    fun `a rejected image is recorded as an error and not as junk`() {
        val bad = photo()
        Mockito.`when`(storage.getBytes(bad.s3Key)).thenReturn(bytesOf(bad))
        Mockito.`when`(client.imageCheck(anyArg())).thenThrow(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, null, "rejected"))

        assertEquals(0, measurer().measure(batchId, token, listOf(bad)))

        Mockito.verify(batches).saveMeasureError(bad.photoId, batchId, token, "BAD_IMAGE")
        verifyNoScoresOrRetry()
        assertEquals(1.0, counted("error"))
    }

    @Test
    fun `a missing original is recorded as an error`() {
        val gone = photo()
        Mockito.`when`(storage.getBytes(gone.s3Key)).thenThrow(IllegalStateException("NoSuchKey"))
        Mockito.`when`(storage.exists(gone.s3Key)).thenReturn(false)

        measurer().measure(batchId, token, listOf(gone))

        Mockito.verify(batches).saveMeasureError(gone.photoId, batchId, token, "NO_FILE")
        Mockito.verifyNoInteractions(client)
        verifyNoScoresOrRetry()
    }

    @Test
    fun `a throttled photo is left for the next tick`() {
        val throttled = photo()
        val storageDown = photo()
        Mockito.`when`(storage.getBytes(throttled.s3Key)).thenReturn(bytesOf(throttled))
        Mockito.`when`(client.imageCheck(anyArg())).thenThrow(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, "THROTTLED", "throttled"))
        Mockito.`when`(storage.getBytes(storageDown.s3Key)).thenThrow(IllegalStateException("timeout"))
        Mockito.`when`(storage.exists(storageDown.s3Key)).thenReturn(true)

        assertEquals(0, measurer().measure(batchId, token, listOf(throttled, storageDown)))

        Mockito.verify(batches).saveMeasureRetry(throttled.photoId, batchId, token, 5)
        Mockito.verify(batches).saveMeasureRetry(storageDown.photoId, batchId, token, 5)
        Mockito.verify(batches, Mockito.never()).saveMeasureError(anyArg(), anyArg(), anyArg(), anyArg())
        assertEquals(2.0, counted("retry"))
    }

    @Test
    fun `no transaction is open during the storage read or the AWS call`() {
        val p = photo()
        val open = mutableMapOf<String, Boolean>()
        Mockito.`when`(storage.getBytes(p.s3Key)).thenAnswer {
            open["read"] = TransactionSynchronizationManager.isActualTransactionActive()
            bytesOf(p)
        }
        Mockito.`when`(client.imageCheck(anyArg())).thenAnswer {
            open["aws"] = TransactionSynchronizationManager.isActualTransactionActive()
            ImageCheck(1.0, 1.0, 1.0, 1.0, emptyList())
        }
        Mockito.`when`(batches.saveMeasurement(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(),
            Mockito.anyDouble(), Mockito.anyDouble())).thenAnswer {
            open["write"] = TransactionSynchronizationManager.isActualTransactionActive()
            1
        }

        measurer().measure(batchId, token, listOf(p))

        assertEquals(mapOf("read" to false, "aws" to false, "write" to true), open)
    }

    private fun verifyNoScoresOrRetry() {
        Mockito.verify(batches, Mockito.never()).saveMeasurement(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(),
            anyArg(), anyArg(), Mockito.anyDouble(), Mockito.anyDouble())
        Mockito.verify(batches, Mockito.never()).saveMeasureRetry(anyArg(), anyArg(), anyArg(), Mockito.anyInt())
    }

    private fun <T> anyArg(): T = Mockito.any()
}
