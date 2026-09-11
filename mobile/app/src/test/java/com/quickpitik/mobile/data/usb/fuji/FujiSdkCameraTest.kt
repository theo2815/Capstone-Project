package com.quickpitik.mobile.data.usb.fuji

import com.quickpitik.mobile.data.usb.ptp.CameraSetupException
import com.quickpitik.mobile.data.usb.ptp.PtpDeviceInfo
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayOutputStream

class FujiSdkCameraTest {
    private class FakeSdk(
        var mode: Long = 0x0001, var record: Long = FujiSdk.MEDIAREC_RAWJPEG,
        val buffer: ArrayDeque<Pair<FujiImageInfo, ByteArray>> = ArrayDeque(),
    ) : FujiSdkApi {
        val calls = mutableListOf<String>()
        override fun load() { calls += "load" }
        override fun init() { calls += "init" }
        override fun exit() { calls += "exit" }
        override fun open(fileDescriptor: Int, rawDescriptors: ByteArray) = 7L.also { calls += "open" }
        override fun close(handle: Long) { calls += "close" }
        override fun deviceInfo(handle: Long) = FujiDeviceInfo("FUJIFILM", "X-T5", "S123", "1.0")
        override fun mode(handle: Long) = mode
        override fun mediaRecord(handle: Long) = record
        override fun setMediaRecord(handle: Long, mode: Long) { calls += "setMediaRecord:$mode"; record = mode }
        override fun readImageInfo(handle: Long) = buffer.firstOrNull()?.first ?: FujiImageInfo("", FujiSdk.FORMAT_NONE.toLong(), 0)
        override fun readImage(handle: Long, into: ByteArray): Int {
            val (_, bytes) = buffer.removeFirst(); bytes.copyInto(into); calls += "readImage"; return bytes.size
        }
        override fun deleteImage(handle: Long) { buffer.removeFirst(); calls += "deleteImage" }
    }
    private val info = PtpDeviceInfo(100, 0, 0, "", emptyList(), emptyList(), emptyList(), emptyList(), listOf(0x3801), "FUJIFILM", "X-T5", "1.0", "S123")
    private fun jpeg(size: Int) = FujiImageInfo("DSCF0001.JPG", FujiSdk.FORMAT_JPEG.toLong(), size.toLong()) to ByteArray(size) { 0xff.toByte() }
    private fun raw() = FujiImageInfo("DSCF0001.RAF", FujiSdk.FORMAT_RAW.toLong(), 9) to ByteArray(9)

    @Test fun `jpeg frames come out of the buffer as transient captures and raw frames are skipped`() {
        val sdk = FakeSdk(buffer = ArrayDeque(listOf(raw(), jpeg(4))))
        val camera = FujiSdkCamera(sdk, 7, info, "camera") {}
        assertTrue(camera.capabilities.transientObjects)
        val (handle, meta) = requireNotNull(camera.nextCapture())
        assertEquals(FujiSdk.BUFFER_HANDLE, handle)
        assertEquals("DSCF0001.JPG", meta.filename)
        assertEquals(4L, meta.expectedSize)
        val out = ByteArrayOutputStream()
        assertEquals(4L, camera.readObject(handle, out, 4))
        assertEquals(4, out.size())
        assertNull(camera.nextCapture())
        assertEquals(listOf("deleteImage", "readImage"), sdk.calls)
    }

    @Test fun `start keeps a card copy only when recording was off and refuses movie mode`() {
        val off = FakeSdk(record = FujiSdk.MEDIAREC_OFF)
        FujiSdkCamera(off, 7, info, "camera") {}.also { it.start(); assertNotNull(it.setupWarning) }
        assertEquals(listOf("setMediaRecord:1"), off.calls)
        val kept = FakeSdk(record = FujiSdk.MEDIAREC_RAWJPEG)
        FujiSdkCamera(kept, 7, info, "camera") {}.start()
        assertTrue(kept.calls.isEmpty())
        val movie = FakeSdk(mode = FujiSdk.MODE_MOVIE_FIRST)
        assertThrows(CameraSetupException::class.java) { FujiSdkCamera(movie, 7, info, "camera") {}.start() }
    }

    @Test fun `close releases the sdk then the usb ownership exactly once`() {
        val sdk = FakeSdk()
        var released = 0
        val camera = FujiSdkCamera(sdk, 7, info, "camera") { released++ }
        camera.close(); camera.close()
        assertEquals(listOf("close", "exit"), sdk.calls)
        assertEquals(1, released)
        assertFalse(camera.isUsable)
    }
}
