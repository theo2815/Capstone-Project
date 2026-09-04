package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.StoredObject
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// Spotted 2026-08-14 during V25 smoke: normaliseToJpeg 500'd on a degenerate
// source. Two distinct integer-division floors reach zero —
//   1×1 → getSubimage(cx, cy, 1, 0)      → RasterFormatException
//   4×1 → BufferedImage(1, 0)            → IllegalArgumentException
// — and both fell through to GlobalExceptionHandler's catch-all INTERNAL_ERROR.
class EventCoverServiceTest {

    private lateinit var storageService: StorageService
    private lateinit var service: EventCoverService

    private val eventId = UUID.randomUUID()

    @BeforeEach
    fun setUp() {
        storageService = Mockito.mock(StorageService::class.java)
        Mockito.`when`(storageService.put(anyArg(), anyArg<ByteArray>(), anyArg()))
            .thenReturn(StoredObject("k", 1L, "image/jpeg"))
        service = EventCoverService(storageService)
    }

    @Test
    fun `a 1x1 source is rejected instead of throwing RasterFormatException`() {
        val ex = assertFailsWith<ValidationException> { upload(jpeg(1, 1)) }

        assertEquals(ErrorCodes.VALIDATION_ERROR, ex.code)
        assertEquals("cover", ex.field)
        verifyNothingStored()
    }

    @Test
    fun `a source that clears the crop but not the downscale is also rejected`() {
        // 4×1 survives getSubimage (cropW=1, cropH=1) and dies one line later
        // on outH = 1 * 3 / 4 = 0.
        val ex = assertFailsWith<ValidationException> { upload(jpeg(4, 1)) }

        assertEquals(ErrorCodes.VALIDATION_ERROR, ex.code)
        verifyNothingStored()
    }

    @Test
    fun `a source narrower than the target ratio is rejected`() {
        val ex = assertFailsWith<ValidationException> { upload(jpeg(1, 40)) }

        assertEquals(ErrorCodes.VALIDATION_ERROR, ex.code)
        verifyNothingStored()
    }

    @Test
    fun `the smallest accepted source still normalises`() {
        val key = upload(jpeg(4, 3))

        assertTrue(key.startsWith("events/$eventId/cover/"))
        assertTrue(key.endsWith(".jpg"))
    }

    @Test
    fun `an ordinary source is unaffected`() {
        val key = upload(jpeg(400, 300))

        assertTrue(key.startsWith("events/$eventId/cover/"))
        Mockito.verify(storageService).put(anyArg(), anyArg<ByteArray>(), anyArg())
    }

    private fun upload(bytes: ByteArray) = service.upload(eventId, bytes, "image/jpeg")

    private fun verifyNothingStored() = Mockito.verify(storageService, Mockito.never())
        .put(anyArg(), anyArg<ByteArray>(), anyArg())

    private fun jpeg(width: Int, height: Int): ByteArray {
        val out = ByteArrayOutputStream()
        ImageIO.write(BufferedImage(width, height, BufferedImage.TYPE_INT_RGB), "jpeg", out)
        return out.toByteArray()
    }

    private fun <T> anyArg(): T = Mockito.any()
}
