package com.quickpitik.mobile.data.usb.ptp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * DeviceInfo is the camera's self-description; driver selection reads the
 * vendor extension, the operation set and the model name off it. The string
 * reader must keep the space inside "Canon EOS R6" while dropping the NUL, and
 * every length field is bounded by what is actually in the payload.
 */
class PtpDeviceInfoTest {
    @Test fun `round-trips a Canon description including the space inside the model name`() {
        val info = PtpDeviceInfo.parse(deviceInfo())
        assertEquals(100, info.standardVersion)
        assertEquals(0xBL, info.vendorExtensionId)
        assertEquals(0x64, info.vendorExtensionVersion)
        assertEquals(DESC, info.vendorExtensionDesc)
        assertEquals(OPS, info.operationsSupported)
        assertEquals(EVENTS, info.eventsSupported)
        assertEquals(emptyList<Int>(), info.devicePropertiesSupported)
        assertEquals(listOf(Ptp.OBJECT_FORMAT_EXIF_JPEG), info.imageFormats)
        assertEquals("Canon Inc.", info.manufacturer)
        assertEquals("Canon EOS R6", info.model)
        assertEquals("3-1.8.0", info.deviceVersion)
        assertEquals("", PtpDeviceInfo.parse(deviceInfo(serial = "")).serialNumber)
    }

    @Test fun `a string or capability count that runs past the payload is refused`() {
        val full = deviceInfo()
        assertEquals("Truncated PTP string.", parseError(full.copyOf(full.size - 4)))
        assertEquals("Truncated camera device information.", parseError(full.copyOf(5)))
        for (count in listOf(60_000, -1)) {
            val bytes = full.copyOf()
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(IMAGE_FORMATS_COUNT_OFFSET, count)
            assertEquals("count $count", "Invalid PTP capability count.", parseError(bytes))
        }
    }

    private fun parseError(bytes: ByteArray): String? =
        assertThrows(PtpException::class.java) { PtpDeviceInfo.parse(bytes) }.message

    private fun deviceInfo(serial: String = "0123456789ab"): ByteArray {
        val b = ByteBuffer.allocate(1024).order(ByteOrder.LITTLE_ENDIAN)
        b.putShort(100).putInt(0xB).putShort(0x64)
        b.ptpString(DESC)
        b.putShort(0) // functional mode
        b.u16Array(OPS)
        b.u16Array(EVENTS)
        b.u16Array(emptyList()) // device properties
        b.u16Array(emptyList()) // capture formats
        b.u16Array(listOf(Ptp.OBJECT_FORMAT_EXIF_JPEG))
        b.ptpString("Canon Inc.")
        b.ptpString("Canon EOS R6")
        b.ptpString("3-1.8.0")
        b.ptpString(serial)
        return b.array().copyOf(b.position())
    }

    private fun ByteBuffer.ptpString(s: String) {
        if (s.isEmpty()) { put(0); return }
        put((s.length + 1).toByte())
        s.forEach { putShort(it.code.toShort()) }
        putShort(0)
    }

    private fun ByteBuffer.u16Array(values: List<Int>) {
        putInt(values.size)
        values.forEach { putShort(it.toShort()) }
    }

    private companion object {
        const val DESC = "canon.com: 1.0;"
        val OPS = listOf(Ptp.OP_GET_DEVICE_INFO, Ptp.OP_OPEN_SESSION, Ptp.OP_EOS_SET_REMOTE_MODE)
        val EVENTS = listOf(Ptp.EV_OBJECT_ADDED)
        // header (2+4+2) + desc string + functional mode + operations + events + two empty arrays
        val IMAGE_FORMATS_COUNT_OFFSET = 8 + (1 + 2 * (DESC.length + 1)) + 2 +
            (4 + 2 * OPS.size) + (4 + 2 * EVENTS.size) + 4 + 4
    }
}
