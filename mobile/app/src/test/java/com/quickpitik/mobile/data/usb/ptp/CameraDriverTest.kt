package com.quickpitik.mobile.data.usb.ptp

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class CameraDriverTest {
    private fun info(manufacturer: String, operations: List<Int>) = PtpDeviceInfo(
        100, 6, 100, "MTP", operations, emptyList(), emptyList(), emptyList(), listOf(0x3801),
        manufacturer, "Test camera", "1.0", "serial",
    )

    @Test fun `Microsoft extension ID does not hide Canon EOS capability`() {
        val info = info("Canon", listOf(Ptp.OP_EOS_SET_REMOTE_MODE, Ptp.OP_EOS_SET_EVENT_MODE))
        assertTrue(CameraDrivers.isCanon(0x04a9, info))
        assertFalse(CameraDrivers.isCanon(0x054c, info.copy(manufacturer = "Sony")))
        assertFalse(CameraDrivers.isCanon(0x04a9, info.copy(operationsSupported = emptyList())))
    }

    @Test fun `generic live initialization sends no vendor operations`() {
        val transport = ScriptedPtp()
        GenericPtpCamera(PtpSession(transport), info("Nikon", emptyList()), "camera").startLive()
        assertTrue(transport.writes.isEmpty())
    }

    @Test fun `R10 warmup and card polling quirks are narrow`() {
        assertTrue(CameraDrivers.needsWarmUp(0x04a9, 0x32f8))
        assertFalse(CameraDrivers.needsWarmUp(0x04a9, 0x32f5))
        assertFalse(CameraDrivers.needsWarmUp(0x054c, 0x32f8))
        assertTrue(CameraDrivers.cardPollingOnly("Canon EOS R6"))
        assertFalse(CameraDrivers.cardPollingOnly("Canon EOS R6 Mark II"))
    }

    @Test fun `object identity separates cameras folders and reused handles`() {
        val photo = PtpObjectInfo(0x3801, 100, "IMG_0001.JPG", 1, 10, "20260907T120000")
        assertNotEquals(photo.sourceKey("camera-a", 1), photo.sourceKey("camera-b", 1))
        assertNotEquals(photo.sourceKey("camera-a", 1), photo.copy(parent = 11).sourceKey("camera-a", 1))
        assertNotEquals(photo.sourceKey("camera-a", 1), photo.copy(capturedAt = "20260907T120001").sourceKey("camera-a", 1))
    }

    @Test fun `truncated metadata and hostile capability counts are rejected`() {
        assertThrows(PtpException::class.java) { PtpObjectInfo.parse(ByteArray(12)) }
        val payload = ByteBuffer.allocate(17).order(ByteOrder.LITTLE_ENDIAN)
            .putShort(100).putInt(6).putShort(100).put(0).putShort(0).putInt(Int.MAX_VALUE).array()
        assertThrows(PtpException::class.java) { PtpDeviceInfo.parse(payload) }
    }
}
