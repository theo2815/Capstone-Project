package com.quickpitik.service.photos

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

// POST /public/photos/verify: fingerprint the upload, find the nearest stored
// preview, and answer with attribution only. The threshold split (strong /
// weak / none) is the whole contract; the repository does the Hamming math.
class PhotoVerifyServiceTest {

    private val photoRepository = Mockito.mock(PhotoRepository::class.java)
    private val settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
    private val userRepository = Mockito.mock(UserRepository::class.java)
    private val eventRepository = Mockito.mock(EventRepository::class.java)
    private val service = PhotoVerifyService(photoRepository, settingsRepository, userRepository, eventRepository, maxDistance = 12)

    private val photographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()

    @Test
    fun `a near-exact match is strong and carries photographer and event attribution`() {
        nearest(distance = 2)
        attribution()

        val result = service.verify(jpeg())

        assertEquals(true, result.matched)
        assertEquals("strong", result.confidence)
        assertEquals("Reyes Race Photos", result.photographerName)
        assertEquals("anareyes", result.photographerHandle)
        assertEquals("Cebu Marathon 2026", result.eventName)
        assertEquals(LocalDate.of(2026, 1, 11), result.eventDate)
        assertEquals(2, result.distance)
    }

    @Test
    fun `a match past half the threshold is weak`() {
        nearest(distance = 9)
        attribution(brandName = null)

        val result = service.verify(jpeg())

        assertEquals("weak", result.confidence)
        // No brand name → the account name, same rule as the baked credit.
        assertEquals("Ana Reyes", result.photographerName)
    }

    @Test
    fun `beyond the threshold is no match and leaks nothing`() {
        nearest(distance = 13)
        attribution()

        val result = service.verify(jpeg())

        assertEquals(false, result.matched)
        assertNull(result.confidence)
        assertNull(result.photographerName)
        assertNull(result.eventName)
        assertNull(result.distance)
    }

    @Test
    fun `an empty registry is no match`() {
        Mockito.`when`(photoRepository.findNearestByPhash(Mockito.anyLong())).thenReturn(emptyList())

        assertEquals(false, service.verify(jpeg()).matched)
    }

    @Test
    fun `undecodable bytes are rejected as unsupported media`() {
        val ex = assertFailsWith<ApiException> { service.verify("definitely not an image".toByteArray()) }

        assertEquals(HttpStatus.UNSUPPORTED_MEDIA_TYPE, ex.status)
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun nearest(distance: Long) {
        Mockito.`when`(photoRepository.findNearestByPhash(Mockito.anyLong()))
            .thenReturn(listOf(arrayOf<Any>(photographerId, eventId, distance)))
    }

    private fun attribution(brandName: String? = "Reyes Race Photos") {
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(
            Optional.of(
                PhotographerSettings(
                    userId = photographerId,
                    brandName = brandName,
                    handle = "anareyes",
                    verificationStatus = VerificationStatus.APPROVED,
                ),
            ),
        )
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(
            Optional.of(
                User(
                    id = photographerId,
                    email = "ana@test.local",
                    passwordHash = "\$2a\$12\$stub",
                    name = "Ana Reyes",
                    role = Role.PHOTOGRAPHER,
                ),
            ),
        )
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(
            Optional.of(
                Event(
                    id = eventId,
                    slug = "cebu-marathon-2026",
                    name = "Cebu Marathon 2026",
                    date = LocalDate.of(2026, 1, 11),
                    location = "Cebu City",
                    status = EventStatus.ACTIVE,
                ),
            ),
        )
    }

    private fun jpeg(): ByteArray {
        val img = BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB)
        val g = img.createGraphics()
        try {
            g.color = Color(30, 90, 160)
            g.fillRect(0, 0, 64, 64)
            g.color = Color.WHITE
            g.fillOval(16, 16, 32, 32)
        } finally {
            g.dispose()
        }
        val out = ByteArrayOutputStream()
        ImageIO.write(img, "jpg", out)
        return out.toByteArray()
    }
}
