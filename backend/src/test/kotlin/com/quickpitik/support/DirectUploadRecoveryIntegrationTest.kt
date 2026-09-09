package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.photographer.PhotoUploadService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import java.time.LocalDate
import java.time.ZoneId
import java.util.HexFormat
import java.util.UUID
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO
import kotlin.test.*

class DirectUploadRecoveryIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var uploads: PhotoUploadService
    @Autowired private lateinit var storage: StorageService
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var settings: PhotographerSettingsRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var eventPhotographers: EventPhotographerRepository

    @Test fun `concurrent commits and lost success preserve one original and one marketplace photo`() {
        exerciseConcurrentCommits(sameSlot = true)
    }

    @Test fun `concurrent duplicate slots resolve after rollback and only delete losing objects`() {
        exerciseConcurrentCommits(sameSlot = false)
    }

    private fun exerciseConcurrentCommits(sameSlot: Boolean) {
        val owner = users.save(User(email = "direct-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Direct Upload Test", role = Role.PHOTOGRAPHER)).id
        settings.save(PhotographerSettings(userId = owner, verificationStatus = VerificationStatus.APPROVED))
        val event = events.save(Event(slug = "direct-${UUID.randomUUID()}", name = "Direct Recovery", location = "Cebu",
            date = LocalDate.now(ZoneId.of("Asia/Manila")), status = EventStatus.ACTIVE,
            createdBy = owner, pricingMode = EventPricingMode.FREE, watermarkPolicy = WatermarkPolicy.NONE))
        val bytes = ByteArrayOutputStream().apply {
            ImageIO.write(BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB), "jpg", this)
        }.toByteArray()
        val hash = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))
        val firstId = UUID.randomUUID()
        val ids = List(4) { if (sameSlot) firstId else UUID.randomUUID() }
        fun key(id: UUID) = "events/${event.id}/photos/$id/original.jpg"
        ids.distinct().forEach { storage.put(key(it), bytes, "image/jpeg") }
        val ready = CountDownLatch(ids.size)
        val start = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(ids.size)
        val results = try {
            val calls = ids.map { id -> executor.submit(Callable {
                ready.countDown()
                check(start.await(10, TimeUnit.SECONDS))
                uploads.commitDirectUpload(owner, event.id, id, key(id), hash, bytes.size.toLong(), "image/jpeg")
            }) }
            assertTrue(ready.await(10, TimeUnit.SECONDS))
            start.countDown()
            calls.map { it.get(30, TimeUnit.SECONDS) }
        } finally { executor.shutdownNow() }
        val winner = results.first().id
        assertTrue(results.all { it.id == winner })
        assertEquals(1, events.findById(event.id).orElseThrow().photoCount)
        assertEquals(1, eventPhotographers.findById(EventPhotographerId(event.id, owner)).orElseThrow().photoCount)
        assertContentEquals(bytes, storage.getBytes(key(winner)))
        ids.distinct().filter { it != winner }.forEach { assertFalse(storage.exists(key(it))) }

        // Exercise the real async watermark trigger and runner-visible state with local storage.
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (photos.findById(winner).orElseThrow().status != PhotoStatus.LIVE && System.nanoTime() < deadline) {
            Thread.sleep(50)
        }
        val published = photos.findById(winner).orElseThrow()
        assertEquals(PhotoStatus.LIVE, published.status)
        assertTrue(storage.exists(requireNotNull(published.watermarkS3Key)))
        // An acknowledgement retry remains a read even after the upload window closes.
        events.save(events.findById(event.id).orElseThrow().apply { date = date.minusMonths(2) })
        val retry = uploads.commitDirectUpload(owner, event.id, winner, key(winner), hash, bytes.size.toLong(), "image/jpeg")
        assertEquals(winner, retry.id)
        assertEquals("live", retry.status)
        assertEquals(1, events.findById(event.id).orElseThrow().photoCount)
        assertContentEquals(bytes, storage.getBytes(key(winner)))
    }
}
