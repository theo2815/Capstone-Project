package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotoUploadIntent
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoUploadIntentRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotoUploadedForWatermark
import com.quickpitik.service.photographer.PhotoUploadService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.TestConfiguration
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Import
import org.springframework.mock.web.MockMultipartFile
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.HexFormat
import java.util.UUID
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import javax.imageio.ImageIO
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@Import(DirectUploadRecoveryIntegrationTest.EventCounterConfig::class)
class DirectUploadRecoveryIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var uploads: PhotoUploadService
    @Autowired private lateinit var storage: StorageService
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var settings: PhotographerSettingsRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var intents: PhotoUploadIntentRepository
    @Autowired private lateinit var eventPhotographers: EventPhotographerRepository
    @Autowired private lateinit var registrationEvents: RegistrationEvents

    @BeforeEach
    fun resetRegistrationEvents() = registrationEvents.reset()

    @Test
    fun `concurrent direct slots preserve one original and one marketplace photo`() {
        val fixture = fixture()
        val issued = List(2) { issue(fixture, fixture.bytes) }

        val results = race(issued.map { intent -> Callable {
            uploads.commitDirectUpload(
                fixture.owner,
                fixture.event.id,
                intent.photoId,
                intent.stagingKey,
                fixture.hash,
                fixture.bytes.size.toLong(),
                "image/jpeg",
            )
        } })

        assertOneRegistration(fixture, results.map { it.id })
        val winner = results.first().id
        assertContentEquals(fixture.bytes, storage.getBytes(photos.findById(winner).orElseThrow().s3Key))
        issued.filter { it.photoId != winner }.forEach { loser ->
            assertTrue(storage.exists(loser.originalKey), "Task 4 janitor, not a stale verifier, owns loser cleanup")
        }
    }

    @Test
    fun `direct and multipart races return one winner without double counters`() {
        val fixture = fixture()
        val direct = issue(fixture, fixture.bytes)
        val multipart = MockMultipartFile("file", "photo.jpg", "image/jpeg", fixture.bytes)

        val results = race(
            listOf(
                Callable {
                    uploads.commitDirectUpload(
                        fixture.owner,
                        fixture.event.id,
                        direct.photoId,
                        direct.stagingKey,
                        fixture.hash,
                        fixture.bytes.size.toLong(),
                        "image/jpeg",
                    )
                },
                Callable { uploads.upload(fixture.owner, fixture.event.id, multipart) },
            ),
        )

        assertOneRegistration(fixture, results.map { it.id })
    }

    @Test
    fun `concurrent direct slots in different events register one winner`() {
        val first = fixture()
        val secondEvent = events.saveAndFlush(
            Event(
                slug = "direct-${UUID.randomUUID()}",
                name = "Direct Recovery Two",
                location = "Cebu",
                date = first.event.date,
                status = EventStatus.ACTIVE,
                createdBy = first.owner,
                pricingMode = EventPricingMode.FREE,
                watermarkPolicy = WatermarkPolicy.NONE,
            ),
        )
        val second = first.copy(event = secondEvent)
        val issued = listOf(issue(first, first.bytes), issue(second, second.bytes))

        val results = race(issued.map { intent -> Callable {
            runCatching {
                uploads.commitDirectUpload(
                    first.owner,
                    intent.eventId,
                    intent.photoId,
                    intent.stagingKey,
                    first.hash,
                    first.bytes.size.toLong(),
                    "image/jpeg",
                )
            }
        } })

        assertEquals(1, results.count { it.isSuccess })
        val conflict = results.single { it.isFailure }.exceptionOrNull() as ConflictException
        assertEquals("PHOTO_DUPLICATE_DIFFERENT_EVENT", conflict.code)
        assertEquals(
            1,
            events.findById(first.event.id).orElseThrow().photoCount +
                events.findById(second.event.id).orElseThrow().photoCount,
        )
        assertEquals(first.owner, photos.findFirstByPhotographerIdAndContentHash(first.owner, first.hash)?.photographerId)
    }

    @Test
    fun `lost acknowledgement after closure returns the immutable original despite staging overwrite`() {
        val fixture = fixture()
        val intent = issue(fixture, fixture.bytes)
        val first = uploads.commitDirectUpload(
            fixture.owner,
            fixture.event.id,
            intent.photoId,
            intent.stagingKey,
            fixture.hash,
            fixture.bytes.size.toLong(),
            "image/jpeg",
        )
        val replacement = jpeg(rgb = 0x00ff00)
        storage.put(intent.stagingKey, replacement, "image/jpeg")
        events.saveAndFlush(events.findById(fixture.event.id).orElseThrow().apply { date = date.minusMonths(2) })

        val retry = uploads.commitDirectUpload(
            fixture.owner,
            fixture.event.id,
            intent.photoId,
            intent.stagingKey,
            fixture.hash,
            fixture.bytes.size.toLong(),
            "image/jpeg",
        )

        assertEquals(first.id, retry.id)
        assertEquals(1, events.findById(fixture.event.id).orElseThrow().photoCount)
        assertContentEquals(fixture.bytes, storage.getBytes(intent.originalKey))
        assertFalse(replacement.contentEquals(storage.getBytes(intent.originalKey)))
    }

    private fun issue(fixture: Fixture, bytes: ByteArray): PhotoUploadIntent {
        val id = UUID.randomUUID()
        val now = OffsetDateTime.now()
        val intent = intents.saveAndFlush(
            PhotoUploadIntent(
                photoId = id,
                photographerId = fixture.owner,
                eventId = fixture.event.id,
                requestId = UUID.randomUUID(),
                contentHash = fixture.hash,
                contentType = "image/jpeg",
                sizeBytes = bytes.size.toLong(),
                stagingKey = "staging/photo-uploads/${fixture.owner}/${fixture.event.id}/$id/original",
                originalKey = "events/${fixture.event.id}/photos/$id/original.jpg",
                createdAt = now,
                putExpiresAt = now.plusMinutes(15),
                expiresAt = now.plusHours(24),
            ),
        )
        storage.put(intent.stagingKey, bytes, intent.contentType)
        return intent
    }

    private fun fixture(): Fixture {
        val owner = users.saveAndFlush(
            User(
                email = "direct-${UUID.randomUUID()}@test.local",
                passwordHash = "unused",
                name = "Direct Upload Test",
                role = Role.PHOTOGRAPHER,
            ),
        ).id
        settings.saveAndFlush(
            PhotographerSettings(userId = owner, verificationStatus = VerificationStatus.APPROVED),
        )
        val event = events.saveAndFlush(
            Event(
                slug = "direct-${UUID.randomUUID()}",
                name = "Direct Recovery",
                location = "Cebu",
                date = LocalDate.now(ZoneId.of("Asia/Manila")),
                status = EventStatus.ACTIVE,
                createdBy = owner,
                pricingMode = EventPricingMode.FREE,
                watermarkPolicy = WatermarkPolicy.NONE,
            ),
        )
        val bytes = jpeg()
        val hash = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))
        return Fixture(owner, event, bytes, hash)
    }

    private fun <T> race(calls: List<Callable<T>>): List<T> {
        val ready = CountDownLatch(calls.size)
        val start = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(calls.size)
        return try {
            val futures = calls.map { call -> executor.submit(Callable {
                ready.countDown()
                check(start.await(10, TimeUnit.SECONDS))
                call.call()
            }) }
            assertTrue(ready.await(10, TimeUnit.SECONDS))
            start.countDown()
            futures.map { it.get(30, TimeUnit.SECONDS) }
        } finally {
            executor.shutdownNow()
        }
    }

    private fun assertOneRegistration(fixture: Fixture, photoIds: List<UUID>) {
        val winner = photoIds.first()
        assertTrue(photoIds.all { it == winner })
        assertEquals(1, events.findById(fixture.event.id).orElseThrow().photoCount)
        assertEquals(
            1,
            eventPhotographers.findById(EventPhotographerId(fixture.event.id, fixture.owner)).orElseThrow().photoCount,
        )
        assertEquals(1, registrationEvents.count(fixture.event.id))
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (photos.findById(winner).orElseThrow().status != PhotoStatus.LIVE && System.nanoTime() < deadline) {
            Thread.sleep(50)
        }
        assertEquals(PhotoStatus.LIVE, photos.findById(winner).orElseThrow().status)
    }

    private fun jpeg(rgb: Int = 0x123456): ByteArray = ByteArrayOutputStream().use { output ->
        val image = BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB)
        image.graphics.apply {
            color = java.awt.Color(rgb)
            fillRect(0, 0, image.width, image.height)
            dispose()
        }
        check(ImageIO.write(image, "jpg", output))
        output.toByteArray()
    }

    private data class Fixture(
        val owner: UUID,
        val event: Event,
        val bytes: ByteArray,
        val hash: String,
    )

    @TestConfiguration(proxyBeanMethods = false)
    class EventCounterConfig {
        @Bean fun registrationEvents() = RegistrationEvents()
    }

    class RegistrationEvents {
        private val counts = ConcurrentHashMap<UUID, AtomicInteger>()

        @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
        fun recorded(event: PhotoUploadedForWatermark) {
            counts.computeIfAbsent(event.eventId) { AtomicInteger() }.incrementAndGet()
        }

        fun count(eventId: UUID): Int = counts[eventId]?.get() ?: 0
        fun reset() = counts.clear()
    }
}
