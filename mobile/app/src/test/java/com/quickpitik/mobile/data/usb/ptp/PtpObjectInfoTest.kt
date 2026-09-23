package com.quickpitik.mobile.data.usb.ptp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The ObjectInfo dataset feeds three decisions on every card import: which
 * objects are JPEGs, how many bytes to expect from GetObject, and the dedupe /
 * card-swap key. A wrong offset here truncates a photo or re-imports a card.
 * (CameraDriverTest already covers the 12-byte payload and hostile counts.)
 */
class PtpObjectInfoTest {
    @Test fun `parses the fields the import path relies on`() {
        val info = PtpObjectInfo.parse(objectInfo())
        assertEquals(0x10001L, info.storageId)
        assertEquals(Ptp.OBJECT_FORMAT_EXIF_JPEG, info.format)
        assertEquals(4_000_000L, info.compressedSize)
        assertEquals(0x6008L, info.parent)
        assertEquals("IMG_0001.JPG", info.filename)
        assertEquals("20260911T101500", info.capturedAt)
        assertEquals(4_000_000L, info.expectedSize)
        assertTrue(info.isJpeg)
        assertFalse(info.isAssociation)
    }

    @Test fun `capture date is optional and a filename count past the payload is refused`() {
        assertEquals("", PtpObjectInfo.parse(objectInfo(capturedAt = null)).capturedAt)
        val bytes = objectInfo().also { it[52] = 0xFF.toByte() }
        val e = assertThrows(PtpException::class.java) { PtpObjectInfo.parse(bytes) }
        assertEquals("Truncated camera object string.", e.message)
    }

    @Test fun `jpeg detection accepts the exif format code or a jpg suffix, nothing else`() {
        assertTrue(PtpObjectInfo(Ptp.OBJECT_FORMAT_EXIF_JPEG, 1, "X.CR3").isJpeg)
        assertTrue(PtpObjectInfo(0x3000, 1, "a.jpg").isJpeg)
        assertTrue(PtpObjectInfo(0x3000, 1, "a.JPEG").isJpeg)
        assertFalse(PtpObjectInfo(0x3000, 1, "a.CR3").isJpeg)
        val folder = PtpObjectInfo(Ptp.OBJECT_FORMAT_ASSOCIATION, 0, "DCIM")
        assertTrue(folder.isAssociation)
        assertFalse(folder.isJpeg)
    }

    @Test fun `expected size excludes zero and the unknown-size sentinel`() {
        assertNull(PtpObjectInfo(Ptp.OBJECT_FORMAT_EXIF_JPEG, 0, "a.jpg").expectedSize)
        assertNull(PtpObjectInfo.parse(objectInfo(size = 0xffffffffL)).expectedSize)
        assertEquals(1L, PtpObjectInfo(Ptp.OBJECT_FORMAT_EXIF_JPEG, 1, "a.jpg").expectedSize)
        assertEquals(0xfffffffeL, PtpObjectInfo.parse(objectInfo(size = 0xfffffffeL)).expectedSize)
    }

    @Test fun `fingerprint length-prefixes each field so neighbours cannot merge`() {
        assertNotEquals(fingerprint("ab", "c"), fingerprint("a", "bc"))
        assertEquals(fingerprint("a", 1L, "b"), fingerprint("a", 1L, "b"))
    }

    private fun objectInfo(
        storage: Long = 0x10001,
        format: Int = Ptp.OBJECT_FORMAT_EXIF_JPEG,
        size: Long = 4_000_000,
        parent: Long = 0x6008,
        filename: String = "IMG_0001.JPG",
        capturedAt: String? = "20260911T101500",
    ): ByteArray {
        val buf = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN)
        buf.putInt(storage.toInt()).putShort(format.toShort()).putShort(0) // 0: storage, format, protection
        buf.putInt(size.toInt()) // 8
        buf.putShort(0).putInt(0).putInt(0).putInt(0).putInt(0).putInt(0).putInt(0) // 12: thumb + image dims
        buf.putInt(parent.toInt()) // 38
        buf.putShort(0).putInt(0).putInt(0) // 42: association type/desc, sequence
        buf.ptpString(filename) // 52
        capturedAt?.let { buf.ptpString(it) }
        return buf.array().copyOf(buf.position())
    }

    private fun ByteBuffer.ptpString(s: String) {
        put((s.length + 1).toByte())
        s.forEach { putShort(it.code.toShort()) }
        putShort(0)
    }
}
