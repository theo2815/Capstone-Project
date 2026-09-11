package com.quickpitik.mobile.data.usb.fuji

import android.hardware.usb.UsbDevice
import com.quickpitik.mobile.data.usb.ptp.CameraSetupException
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadow.api.Shadow

@RunWith(RobolectricTestRunner::class)
class FujiSdkCamerasTest {
    private class Sdk(private val openError: Long? = null) : FujiSdkApi {
        val calls = mutableListOf<String>()
        override fun load() { calls += "load" }
        override fun init() { calls += "init" }
        override fun exit() { calls += "exit" }
        override fun open(fileDescriptor: Int, rawDescriptors: ByteArray): Long {
            calls += "open:$fileDescriptor"
            openError?.let { throw FujiSdkException(0x1602, it) }
            return 7
        }
        override fun close(handle: Long) { calls += "close" }
        override fun deviceInfo(handle: Long) = FujiDeviceInfo("FUJIFILM", "X-T5", "S123", "1.0")
        override fun mode(handle: Long) = 0x0001L
        override fun mediaRecord(handle: Long) = FujiSdk.MEDIAREC_RAWJPEG
        override fun setMediaRecord(handle: Long, mode: Long) {}
        override fun readImageInfo(handle: Long) = FujiImageInfo("", FujiSdk.FORMAT_NONE.toLong(), 0)
        override fun readImage(handle: Long, into: ByteArray) = 0
        override fun deleteImage(handle: Long) {}
    }
    private val device: UsbDevice = Shadow.newInstanceOf(UsbDevice::class.java)
    private var closedHandles = 0
    private val connect: (UsbDevice) -> UsbHandle? = { UsbHandle(42, byteArrayOf(1, 2)) { closedHandles++ } }

    @Test fun `disabled builds and non-fujifilm devices return null without touching the sdk`() {
        val sdk = Sdk()
        assertNull(FujiSdkCameras.open(device, connect, sdk, enabled = false, vendorId = 0x04cb))
        assertNull(FujiSdkCameras.open(device, connect, sdk, enabled = true, vendorId = 0x04a9))
        assertTrue(sdk.calls.isEmpty())
    }

    @Test fun `a tether-mode body opens through the sdk with identity from its serial`() {
        val sdk = Sdk()
        val camera = requireNotNull(FujiSdkCameras.open(device, connect, sdk, enabled = true, vendorId = 0x04cb))
        assertEquals("X-T5", camera.info.model)
        assertEquals(listOf("load", "init", "open:42"), sdk.calls)
        assertFalse(FujiSdkCameras.ownershipAvailable())
        camera.close()
        assertEquals(1, closedHandles)
        assertTrue(FujiSdkCameras.ownershipAvailable())
    }

    @Test fun `wrong usb mode stops with guidance and an unsupported body falls back to ptp`() {
        val wrongMode = Sdk(openError = FujiSdk.ERR_INVALID_USBMODE)
        val error = assertThrows(CameraSetupException::class.java) {
            FujiSdkCameras.open(device, connect, wrongMode, enabled = true, vendorId = 0x04cb)
        }
        assertTrue(error.message!!.contains("USB TETHER SHOOTING AUTO"))
        assertEquals(listOf("load", "init", "open:42", "exit"), wrongMode.calls)
        assertTrue(FujiSdkCameras.ownershipAvailable())
        val noModule = Sdk(openError = FujiSdk.ERR_NO_MODEL_MODULE)
        assertNull(FujiSdkCameras.open(device, connect, noModule, enabled = true, vendorId = 0x04cb))
        assertTrue(FujiSdkCameras.ownershipAvailable())
        assertEquals(2, closedHandles)
    }
}
