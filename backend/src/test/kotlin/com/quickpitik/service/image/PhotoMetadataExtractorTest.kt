package com.quickpitik.service.image

import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PhotoMetadataExtractorTest {
    @Test
    fun `reads standard EXIF values without inventing an offset`() {
        val fields = javaClass.getResourceAsStream("/photo-metadata/complete-exif.jpg")!!.use {
            PhotoMetadataExtractor().extract(it)
        }
        assertEquals("1/500 s", fields.exposureTime)
        assertEquals("Did not fire", fields.flash)
        assertEquals("Auto", fields.whiteBalance)
        assertNull(fields.takenAtOffset)
    }

    @Test
    fun `rejects negative rational denominators and uses valid XMP aperture`() {
        val fields = javaClass.getResourceAsStream("/photo-metadata/rejected-values.jpg")!!.use {
            PhotoMetadataExtractor().extract(it)
        }
        assertNull(fields.exposureTime)
        assertEquals(5.6, fields.aperture)
    }

    @Test
    fun `invalid EXIF flash falls back to XMP Fired`() {
        val fields = javaClass.getResourceAsStream("/photo-metadata/rejected-values.jpg")!!.use {
            PhotoMetadataExtractor().extract(it)
        }
        assertEquals("Did not fire", fields.flash)
    }

    @Test
    fun `reads rational XMP FNumber from embedded metadata`() {
        val fields = javaClass.getResourceAsStream("/photo-metadata/xmp-payload-limit.jpg")!!.use {
            PhotoMetadataExtractor().extract(it)
        }
        assertEquals(2.8, fields.aperture)
    }

    @Test
    fun `serialized allowlist stays within eight KiB with supplementary text`() {
        val fields = javaClass.getResourceAsStream("/photo-metadata/xmp-payload-limit.jpg")!!.use {
            PhotoMetadataExtractor().extract(it)
        }
        val bytes = jacksonObjectMapper().registerModule(JavaTimeModule()).writeValueAsBytes(fields)
        assertTrue(fields.cameraMake?.isNotBlank() == true)
        assertTrue(fields.author?.contains('"') == true)
        assertTrue(fields.author?.contains('\\') == true)
        assertTrue(fields.copyright?.isNotBlank() == true)
        assertTrue(bytes.size <= 8 * 1024, "serialized metadata was ${bytes.size} bytes")
        assertFalse(String(bytes, Charsets.UTF_8).contains("GPS_SENTINEL_QP_94"))
    }

    @Test
    fun `normalizes fixture vectors and excludes private source tags`() {
        val mapper = jacksonObjectMapper().registerModule(JavaTimeModule())
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS)
            .setDefaultPropertyInclusion(JsonInclude.Include.NON_NULL)
        val evidence = javaClass.getResourceAsStream("/photo-metadata/expected.json")!!.use { mapper.readTree(it) }
        val vectors = evidence["vectors"]
        val originals = StringBuilder()
        vectors.properties().forEach { (filename, expected) ->
            val bytes = javaClass.getResourceAsStream("/photo-metadata/$filename")!!.use { it.readAllBytes() }
            originals.append(String(bytes, Charsets.ISO_8859_1))
            val fields = bytes.inputStream().use { PhotoMetadataExtractor().extract(it) }
            val actual = mapper.valueToTree<JsonNode>(fields)
            assertEquals(expected.fieldNames().asSequence().toSet(), actual.fieldNames().asSequence().toSet(), filename)
            expected.properties().forEach { (name, value) -> assertEquals(value.asText(), actual[name].asText(), "$filename $name") }
            assertEquals(!expected.isEmpty, fields.hasRecordedTags(), filename)
            evidence["sentinels"].forEach { sentinel -> assertFalse(actual.toString().contains(sentinel.asText()), filename) }
        }
        evidence["sentinels"].forEach { sentinel -> assertTrue(originals.contains(sentinel.asText()), sentinel.asText()) }
    }

    @Test
    fun `corrupt image causes parser failure for the caller to handle`() {
        assertFailsWith<Exception> {
            javaClass.getResourceAsStream("/photo-metadata/malformed.jpg")!!.use { PhotoMetadataExtractor().extract(it) }
        }
    }

    @Test
    fun `a QuickPitik preview gives back the photo id from its marker`() {
        val photoId = UUID.randomUUID()
        val preview = JpegXmp.inject(smallJpeg(), JpegXmp.creditPacket("Ana Cruz", "ana", photoId, 2026))
        assertEquals(photoId, PhotoMetadataExtractor().extractWithMarker(preview.inputStream()).xmpPhotoId)
    }

    @Test
    fun `a marker that is not a UUID is ignored`() {
        val photoId = UUID.randomUUID()
        val packet = JpegXmp.creditPacket("Ana Cruz", "ana", photoId, 2026).replace(photoId.toString(), "not-a-uuid")
        val extracted = PhotoMetadataExtractor().extractWithMarker(JpegXmp.inject(smallJpeg(), packet).inputStream())
        assertNull(extracted.xmpPhotoId)
        assertEquals("Ana Cruz", extracted.fields.author)
    }

    @Test
    fun `a photo with no marker has none and the same fields as before`() {
        val bytes = javaClass.getResourceAsStream("/photo-metadata/complete-exif.jpg")!!.use { it.readAllBytes() }
        val extracted = PhotoMetadataExtractor().extractWithMarker(bytes.inputStream())
        assertNull(extracted.xmpPhotoId)
        assertEquals(PhotoMetadataExtractor().extract(bytes.inputStream()), extracted.fields)
    }

    private fun smallJpeg(): ByteArray = ByteArrayOutputStream().also {
        ImageIO.write(BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB), "jpg", it)
    }.toByteArray()
}
