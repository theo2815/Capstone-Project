package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoMetadataRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotoUploadService
import com.quickpitik.service.photographer.PhotoWatermarkService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.mock.web.MockMultipartFile
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.time.LocalDate
import java.time.ZoneId
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PhotoUploadAcceptanceIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var uploadService: PhotoUploadService
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var settings: PhotographerSettingsRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var metadata: PhotoMetadataRepository
    @Autowired lateinit var coverage: EventPhotographerRepository
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var watermark: PhotoWatermarkService

    @Test
    fun `racing first uploads and retries keep one first accepted metadata row and counters`() {
        val owner = users.saveAndFlush(User(email = "accept-${UUID.randomUUID()}@test.local",
            passwordHash = "stub", name = "Photographer", role = Role.PHOTOGRAPHER))
        settings.saveAndFlush(PhotographerSettings(userId = owner.id,
            verificationStatus = VerificationStatus.APPROVED))
        val event = newEvent(owner.id)
        Mockito.`when`(storage.presignedGetUrl(Mockito.anyString(), anyArg())).thenReturn("https://thumb")
        val bytes = ByteArrayOutputStream().use { out ->
            check(ImageIO.write(BufferedImage(32, 24, BufferedImage.TYPE_INT_RGB), "jpg", out))
            out.toByteArray()
        }
        fun file(name: String) = MockMultipartFile("file", name, "image/jpeg", bytes)
        val start = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        try {
            val first = pool.submit<UUID> { start.await(); uploadService.upload(owner.id, event.id, file("first.jpg")).id }
            val second = pool.submit<UUID> { start.await(); uploadService.upload(owner.id, event.id, file("second.jpg")).id }
            start.countDown()
            val id = first.get(15, TimeUnit.SECONDS)
            assertEquals(id, second.get(15, TimeUnit.SECONDS))
            val firstName = metadata.findById(id).orElseThrow().originalFilename
            assertTrue(firstName in setOf("first.jpg", "second.jpg"))
            assertEquals(bytes.size.toLong(), metadata.findById(id).orElseThrow().originalSizeBytes)

            assertEquals(id, uploadService.upload(owner.id, event.id, file("retry.jpg")).id)
            assertEquals(firstName, metadata.findById(id).orElseThrow().originalFilename)
            assertFailsWith<ConflictException> {
                uploadService.upload(owner.id, newEvent(owner.id).id, file("other-event.jpg"))
            }
            assertEquals(1, photos.findAll().count { it.photographerId == owner.id })
            assertEquals(1, metadata.findAll().count { it.photoId == id })
            assertEquals(1, events.findById(event.id).orElseThrow().photoCount)
            val row = coverage.findById(EventPhotographerId(event.id, owner.id)).orElseThrow()
            assertEquals(1, row.photoCount)
            assertEquals(0, row.salesCount)
        } finally {
            pool.shutdownNow()
        }
    }

    private fun newEvent(ownerId: UUID) = events.saveAndFlush(Event(
        slug = "accept-${UUID.randomUUID()}", name = "Acceptance Test",
        date = LocalDate.now(ZoneId.of("Asia/Manila")), location = "Cebu",
        status = EventStatus.ACTIVE, createdBy = ownerId, watermarkPolicy = WatermarkPolicy.NONE,
    ))

    private fun <T> anyArg(): T = Mockito.any()
}
