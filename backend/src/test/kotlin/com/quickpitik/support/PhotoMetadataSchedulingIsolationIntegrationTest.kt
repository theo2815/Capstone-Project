package com.quickpitik.support

import com.quickpitik.config.AsyncConfig
import com.quickpitik.config.PhotoMetadataProperties
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.image.PhotoMetadataExtractor
import com.quickpitik.service.photographer.PhotoMetadataService
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.StoredObject
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.context.annotation.AnnotationConfigApplicationContext
import org.springframework.boot.test.context.TestConfiguration
import org.springframework.context.annotation.Import
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.scheduling.TaskScheduler
import org.springframework.scheduling.annotation.EnableScheduling
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.LocalDate
import java.util.UUID
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.function.Supplier
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PhotoMetadataSchedulingIsolationIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var metadata: PhotoMetadataRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var properties: PhotoMetadataProperties
    @Autowired private lateinit var clock: Clock
    @Autowired private lateinit var transactions: TransactionTemplate
    @Autowired private lateinit var service: PhotoMetadataService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired @Qualifier("watermarkProcessing") private lateinit var watermarkPool: ThreadPoolTaskExecutor
    @Autowired @Qualifier("metadataProcessing") private lateinit var metadataPool: ThreadPoolTaskExecutor
    @MockitoBean private lateinit var storage: StorageService
    @MockitoBean private lateinit var extractor: PhotoMetadataExtractor

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun clearPriorMetadataFixtures() {
        awaitIdle(metadataPool)
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while ((watermarkPool.activeCount != 0 || watermarkPool.threadPoolExecutor.queue.isNotEmpty()) && System.nanoTime() < deadline) Thread.sleep(20)
        assertTrue(watermarkPool.activeCount == 0 && watermarkPool.threadPoolExecutor.queue.isEmpty())
        jdbc.update("DELETE FROM photo_metadata")
    }

    @Test
    fun `another test context cannot schedule recovery of this fixture but manual recovery still works`() {
        val event = events.save(Event(slug = "scheduler-${UUID.randomUUID()}", name = "Run", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        val photo = photos.save(Photo(eventId = event.id, s3Key = "original-${UUID.randomUUID()}", pricePhp = BigDecimal.ONE, status = PhotoStatus.LIVE))
        metadata.save(PhotoMetadata(photoId = photo.id, originalFilename = "camera.jpg", originalSizeBytes = 42))
        val oversized = 25L * 1024 * 1024 + 1
        Mockito.`when`(storage.metadata(photo.s3Key)).thenReturn(StoredObject(photo.s3Key, oversized, "image/jpeg"))
        jdbc.update("UPDATE photos SET uploaded_at = now() - interval '6 minutes' WHERE id = ?", photo.id)

        val foreignPool = AsyncConfig().metadataProcessingExecutor()
        val foreignStorage = Mockito.mock(StorageService::class.java) // No object in the other context's storage.
        val foreignWorker = PhotoMetadataService(metadata, photos, coverage, extractor, foreignStorage, watermarkPool, properties, transactions, clock, foreignPool)
        val scheduled = mutableListOf<Runnable>()
        val scheduler = Mockito.mock(TaskScheduler::class.java) { invocation ->
            when {
                invocation.method.name.startsWith("schedule") -> {
                    scheduled += invocation.getArgument<Runnable>(0)
                    Mockito.mock(ScheduledFuture::class.java)
                }
                invocation.method.name == "getClock" -> clock
                else -> Mockito.RETURNS_DEFAULTS.answer(invocation)
            }
        }
        val tickProbe = TickProbe()

        try {
            AnnotationConfigApplicationContext().use { foreignContext ->
                foreignContext.register(ForeignContext::class.java)
                foreignContext.registerBean("taskScheduler", TaskScheduler::class.java, Supplier { scheduler })
                foreignContext.registerBean(PhotoMetadataService::class.java, Supplier { foreignWorker })
                foreignContext.registerBean(TickProbe::class.java, Supplier { tickProbe })
                foreignContext.refresh()
                scheduled.forEach(Runnable::run) // A deterministic scheduler tick, without waiting for a timer.
                awaitIdle(foreignPool)
                assertEquals(1, tickProbe.ticks, "The controlled scheduler must actually fire its registered tasks")

                val untouched = metadata.findById(photo.id).orElseThrow()
                assertEquals(PhotoMetadataState.PENDING, untouched.state, "Foreign scheduler changed fixture ${photo.id}, size=${untouched.originalSizeBytes}")
                assertEquals(0, metadata.findById(photo.id).orElseThrow().attempts)
                Mockito.verifyNoInteractions(foreignStorage)
            }
            service.recoverDue()
            awaitIdle(metadataPool)
            val row = metadata.findById(photo.id).orElseThrow()
            assertEquals(PhotoMetadataState.UNAVAILABLE, row.state)
            assertEquals(1, row.attempts)
            assertEquals(oversized, row.originalSizeBytes)
            Mockito.verify(storage, Mockito.never()).open(anyArg())
        } finally {
            foreignPool.shutdown()
        }
    }

    private fun awaitIdle(pool: ThreadPoolTaskExecutor) {
        // The single worker processes this barrier after every previously submitted recovery task.
        pool.submit {}.get(10, TimeUnit.SECONDS)
    }

    @TestConfiguration(proxyBeanMethods = false)
    @EnableScheduling
    @Import(MetadataRecoveryTestIsolation::class)
    class ForeignContext

    class TickProbe {
        var ticks = 0
        @Scheduled(fixedDelay = 60000)
        fun tick() { ticks++ }
    }
}
