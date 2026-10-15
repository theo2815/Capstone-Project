package com.quickpitik.service.photos

import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.image.PerceptualHash
import com.quickpitik.service.photographer.WatermarkCredit
import com.quickpitik.service.photographer.WatermarkService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

// Security audit 2026-10-14, photos/public-verify-full-decode-80mp.
// The golden values were taken from the code at 7def0878 (SA-11) before any change.
class PhotoVerifyFingerprintRegressionTest {
    companion object {
        const val STORED_1280 = 0x338C8C3373738D8CL
        const val STORED_4000 = 0x338C8C333373CD8CL
        const val VERIFY_1280 = 0x338C8C3373738D8CL
    }

    private val photoRepository = Mockito.mock(PhotoRepository::class.java)
    private val service = PhotoVerifyService(
        photoRepository,
        Mockito.mock(PhotographerSettingsRepository::class.java),
        Mockito.mock(UserRepository::class.java),
        Mockito.mock(EventRepository::class.java),
        maxDistance = 12,
    )
    private val photographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()
    private var storedForCurrentFixture = 0L

    private fun fixture(name: String): ByteArray =
        assertNotNull(javaClass.getResourceAsStream("/photo-verify/$name")).use { it.readBytes() }

    private val small = fixture("scene-1280x853.jpg")
    private val large = fixture("scene-4000x2667.jpg")

    @BeforeEach
    fun setUp() {
        Mockito.`when`(photoRepository.findNearestByPhash(Mockito.anyLong())).thenAnswer { call ->
            val given = call.getArgument<Long>(0)
            listOf(arrayOf<Any>(photographerId, eventId, PerceptualHash.distance(given, storedForCurrentFixture)))
        }
    }

    private fun stored(bytes: ByteArray): Long = WatermarkService("test-seed-secret").processThumbnail(
        bytes, null, WatermarkCredit(name = "Fixture", handle = "fixture", photoId = UUID(0, 1)), platformMark = false,
    ).phashClean

    @Test
    fun `the upload path still stores the frozen hash for both fixtures`() {
        assertEquals(STORED_1280, stored(small))
        assertEquals(STORED_4000, stored(large))
    }

    @Test
    fun `a preview-size copy hashes exactly as before`() {
        val decoded = assertNotNull(service.decodeForHash(small))

        assertEquals(VERIFY_1280, PerceptualHash.of(decoded))
    }

    @Test
    fun `both fixtures still verify as a strong match`() {
        for ((bytes, storedHash) in listOf(small to STORED_1280, large to STORED_4000)) {
            storedForCurrentFixture = storedHash

            val result = service.verify(bytes)

            assertTrue(result.matched)
            assertEquals("strong", result.confidence)
        }
    }

    @Disabled("SA-11 reproduction; SA-12 removes this line")
    @Test
    fun `verify never decodes a huge image at full size`() {
        val huge = BufferedImage(8000, 6000, BufferedImage.TYPE_BYTE_BINARY)       // 48 MP, under the 80 MP cap; ~6 MB raster in this test
        val png = ByteArrayOutputStream().also { ImageIO.write(huge, "png", it) }.toByteArray()
        val decoded = assertNotNull(service.decodeForHash(png))
        assertTrue(maxOf(decoded.width, decoded.height) < 2 * 1280, "decoded ${decoded.width}x${decoded.height}")
    }
}
