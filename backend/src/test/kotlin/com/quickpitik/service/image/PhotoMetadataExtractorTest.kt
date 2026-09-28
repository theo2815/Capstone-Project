package com.quickpitik.service.image

import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import org.junit.jupiter.api.Test
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
}
