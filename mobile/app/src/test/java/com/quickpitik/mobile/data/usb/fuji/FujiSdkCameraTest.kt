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
        var readInfoError: Long? = null,
        var mediaRecordError: Long? = null,
    ) : FujiSdkApi {
        val calls = mutableListOf<String>()
        override fun load() { calls += "load" }
        override fun init() { calls += "init" }
        override fun exit() { calls += "exit" }
        override fun open(fileDescriptor: Int, rawDescriptors: ByteArray) = 7L.also { calls += "open" }
        override fun close(handle: Long) { calls += "close" }
        override fun deviceInfo(handle: Long) = FujiDeviceInfo("FUJIFILM", "X-T5", "S123", "1.0")
        override fun mode(handle: Long) = mode
        override fun mediaRecord(handle: Long): Long {
            mediaRecordError?.let { throw FujiSdkException(0x1025, it) }
            return record
        }
        override fun readImageInfo(handle: Long): FujiImageInfo {
            readInfoError?.let { throw FujiSdkException(0x1501, it) }
            return buffer.firstOrNull()?.first ?: FujiImageInfo("", FujiSdk.FORMAT_NONE.toLong(), 0)
        }
        override fun readImage(handle: Long, into: ByteArray): Int {
            val (_, bytes) = buffer.removeFirst(); bytes.copyInto(into); calls += "readImage"; return bytes.size
        }
        override fun deleteImage(handle: Long) { buffer.removeFirst(); calls += "deleteImage" }
    }
    private val info = PtpDeviceInfo(100, 0, 0, "", emptyList(), emptyList(), emptyList(), emptyList(), listOf(0x3801), "FUJIFILM", "X-T5", "1.0", "S123")
    private fun jpeg(size: Int) = FujiImageInfo("DSCF0001.JPG", FujiSdk.FORMAT_JPEG.toLong(), size.toLong()) to ByteArray(size) { 0xff.toByte() }
    private fun raw() = FujiImageInfo("DSCF0001.RAF", FujiSdk.FORMAT_RAW.toLong(), 9) to ByteArray(9)
    private fun unknown() = FujiImageInfo("DSCF0001.BIN", 0x1f, 9) to ByteArray(9)

    @Test fun `combined mode transfers jpeg and discards only recognized alternate formats`() {
        val sdk = FakeSdk()
        val camera = FujiSdkCamera(sdk, 7, info, "camera") {}
        camera.start()
        sdk.buffer.addAll(listOf(raw(), jpeg(4)))
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

    @Test fun `start rejects every unverified recording mode without consuming the buffer`() {
        for ((record, frame) in listOf(
            FujiSdk.MEDIAREC_OFF to raw(),
            FujiSdk.MEDIAREC_RAW to jpeg(4),
            FujiSdk.MEDIAREC_JPEG to raw(),
            0x99L to unknown(),
        )) {
            val sdk = FakeSdk(record = record, buffer = ArrayDeque(listOf(frame)))
            assertThrows(CameraSetupException::class.java) { FujiSdkCamera(sdk, 7, info, "camera") {}.start() }
            assertTrue(sdk.calls.isEmpty())
            assertEquals(1, sdk.buffer.size)
        }
    }

    @Test fun `start rejects movie mode old buffers and an unverifiable busy buffer`() {
        val movie = FakeSdk(mode = FujiSdk.MODE_MOVIE_FIRST)
        assertThrows(CameraSetupException::class.java) { FujiSdkCamera(movie, 7, info, "camera") {}.start() }

        val pending = FakeSdk(buffer = ArrayDeque(listOf(jpeg(4))))
        assertThrows(CameraSetupException::class.java) { FujiSdkCamera(pending, 7, info, "camera") {}.start() }
        assertTrue(pending.calls.isEmpty())
        assertEquals(1, pending.buffer.size)

        val busy = FakeSdk(readInfoError = FujiSdk.ERR_BUSY)
        assertThrows(CameraSetupException::class.java) { FujiSdkCamera(busy, 7, info, "camera") {}.start() }
        assertTrue(busy.calls.isEmpty())

        val unreadableMode = FakeSdk(mediaRecordError = FujiSdk.ERR_RUNNING_OTHER_FUNCTION)
        assertThrows(CameraSetupException::class.java) { FujiSdkCamera(unreadableMode, 7, info, "camera") {}.start() }
        assertTrue(unreadableMode.calls.isEmpty())
    }

    @Test fun `mode changes stop before next capture or jpeg consumption`() {
        val beforePoll = FakeSdk()
        val pollingCamera = FujiSdkCamera(beforePoll, 7, info, "camera") {}
        pollingCamera.start()
        beforePoll.buffer.add(jpeg(4))
        beforePoll.record = FujiSdk.MEDIAREC_JPEG
        assertThrows(CameraSetupException::class.java) { pollingCamera.nextCapture() }
        assertTrue(beforePoll.calls.isEmpty())

        val beforeRead = FakeSdk()
        val readingCamera = FujiSdkCamera(beforeRead, 7, info, "camera") {}
        readingCamera.start()
        beforeRead.buffer.add(jpeg(4))
        val (handle, meta) = requireNotNull(readingCamera.nextCapture())
        beforeRead.record = FujiSdk.MEDIAREC_RAW
        assertThrows(CameraSetupException::class.java) {
            readingCamera.readObject(handle, ByteArrayOutputStream(), meta.expectedSize)
        }
        assertTrue(beforeRead.calls.isEmpty())
        assertEquals(1, beforeRead.buffer.size)
    }

    @Test fun `unknown frame formats are preserved and rejected`() {
        val sdk = FakeSdk()
        val camera = FujiSdkCamera(sdk, 7, info, "camera") {}
        camera.start()
        sdk.buffer.add(unknown())
        assertThrows(CameraSetupException::class.java) { camera.nextCapture() }
        assertTrue(sdk.calls.isEmpty())
        assertEquals(1, sdk.buffer.size)
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
