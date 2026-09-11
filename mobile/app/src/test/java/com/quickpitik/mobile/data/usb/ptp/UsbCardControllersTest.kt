package com.quickpitik.mobile.data.usb.ptp

import android.content.Context
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.LowCameraStorage
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadow.api.Shadow
import java.io.ByteArrayOutputStream
import java.io.OutputStream

/**
 * The two short-session halves of "Import from camera" over a fake body:
 * browse lists only the JPEGs and always closes the session (an open session
 * freezes the R6's shutter); import re-verifies each photo against the card
 * before any bytes move (a swapped card must not land in the active event),
 * isolates per-photo failures, spends nothing when the phone is full, and
 * writes off the rest of a run once the session is dead.
 */
@RunWith(RobolectricTestRunner::class)
class UsbCardControllersTest {
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val manager = context.getSystemService(Context.USB_SERVICE) as UsbManager
    private val device: UsbDevice = Shadow.newInstanceOf(UsbDevice::class.java)
    private val info = PtpDeviceInfo(100, 0, 0, "", emptyList(), emptyList(), emptyList(), emptyList(), listOf(0x3801), "Canon", "EOS R6", "1.0", "S1")

    private open inner class Camera(val objects: Map<Long, PtpObjectInfo>) : GenericPtpCamera(PtpSession(ScriptedPtp()), info, "cam-a") {
        var closes = 0
        val reads = mutableListOf<Long>()
        override fun handles() = objects.keys.toList()
        override fun objectInfo(handle: Long) = objects[handle] ?: throw PtpException("No object $handle on the card.")
        override fun readObject(handle: Long, sink: OutputStream, expectedSize: Long?): Long {
            reads += handle
            sink.write(byteArrayOf(0xff.toByte(), 0xd8.toByte(), handle.toByte()))
            return 3
        }
        override fun close() { closes++ }
    }

    private fun jpeg(name: String, size: Long = 100) = PtpObjectInfo(Ptp.OBJECT_FORMAT_EXIF_JPEG, size, name, 0x10001, 0x6008, "20260911T100000")
    private fun cardPhoto(camera: Camera, handle: Long) = camera.objects.getValue(handle).let {
        CardPhoto(handle, it.filename, it.compressedSize, cameraKey = camera.cameraKey, objectInfo = it)
    }

    private fun browse(camera: Camera): List<UsbCardBrowseController.Progress> {
        val seen = mutableListOf<UsbCardBrowseController.Progress>()
        runBlocking { UsbCardBrowseController(manager, { device }, { camera }).browse { seen += it } }
        return seen
    }

    private class ImportRun(
        val progress: MutableList<UsbCardImportController.Progress> = mutableListOf(),
        val pulled: MutableMap<Long, ByteArray> = mutableMapOf(),
        val failed: MutableMap<Long, String> = mutableMapOf(),
    )

    private fun import(
        camera: Camera,
        photos: List<CardPhoto>,
        persist: (CardPhoto, ByteArray) -> Boolean = { _, _ -> true },
        deviceProvider: () -> UsbDevice? = { device },
    ): ImportRun {
        val run = ImportRun()
        runBlocking {
            UsbCardImportController(manager, deviceProvider, { camera }).import(
                photos,
                emit = { run.progress += it },
                onPulled = { photo, writeTo ->
                    val bytes = ByteArrayOutputStream().also(writeTo).toByteArray()
                    run.pulled[photo.handle] = bytes
                    persist(photo, bytes)
                },
                onFailed = { photo, message -> run.failed[photo.handle] = message },
            )
        }
        return run
    }

    @Test fun `browse lists only JPEGs with a fallback name, reports each handle, and closes the camera`() {
        val camera = Camera(mapOf(
            1L to jpeg("IMG_0001.JPG"),
            2L to PtpObjectInfo(Ptp.OBJECT_FORMAT_ASSOCIATION, 0, "DCIM"),
            3L to PtpObjectInfo(0xB101, 20_000_000, "IMG_0003.CR3"),
            4L to jpeg(""),
        ))
        val seen = browse(camera)
        val done = seen.filterIsInstance<UsbCardBrowseController.Progress.Done>().single()
        assertEquals(listOf(1L, 4L), done.photos.map { it.handle })
        assertEquals("IMG_00000004.jpg", done.photos[1].filename)
        assertEquals("cam-a", done.photos[0].cameraKey)
        assertEquals(4, seen.count { it is UsbCardBrowseController.Progress.Scanning })
        assertEquals(UsbCardBrowseController.Progress.Scanning(4, 4), seen.last { it is UsbCardBrowseController.Progress.Scanning })
        assertTrue(seen.none { it is UsbCardBrowseController.Progress.Thumbnails }) // this body advertises no GetThumb
        assertEquals(1, camera.closes)
    }

    @Test fun `browse on an empty card is done at once, and thumbnail misses still emit the final list`() {
        assertEquals(
            listOf(UsbCardBrowseController.Progress.Opening, UsbCardBrowseController.Progress.Done(emptyList())),
            browse(Camera(emptyMap())),
        )

        val thumbs = object : Camera(mapOf(1L to jpeg("IMG_0001.JPG"))) {
            override val capabilities get() = super.capabilities.copy(thumbnails = true)
        }
        val seen = browse(thumbs)
        val final = seen.last() as UsbCardBrowseController.Progress.Thumbnails
        assertEquals(1, final.photos.size)
        assertNull(final.photos.single().thumbnailBytes) // GetThumb failed on this body: placeholder tile
        assertEquals(1, thumbs.closes)
    }

    @Test fun `browse turns a read failure mid-scan into a session failure, not a partial list`() {
        val camera = object : Camera(mapOf(1L to jpeg("IMG_0001.JPG"), 2L to jpeg("IMG_0002.JPG"))) {
            override fun objectInfo(handle: Long) = if (handle == 2L) throw PtpException("short read") else super.objectInfo(handle)
        }
        val seen = browse(camera)
        assertTrue(seen.last() is UsbCardBrowseController.Progress.Failed)
        assertTrue(seen.none { it is UsbCardBrowseController.Progress.Done })
        assertEquals(1, camera.closes)
    }

    @Test fun `import pulls the selection in order, tallies per photo, and always closes the camera`() {
        val camera = Camera(mapOf(1L to jpeg("A.JPG"), 2L to jpeg("B.JPG"), 3L to jpeg("C.JPG")))
        val photos = listOf(1L, 2L, 3L).map { cardPhoto(camera, it) }
        val run = import(camera, photos, persist = { photo, _ -> photo.handle != 2L })

        assertEquals(listOf(1L, 2L, 3L), camera.reads)
        assertTrue(run.pulled.getValue(1L).contentEquals(byteArrayOf(0xff.toByte(), 0xd8.toByte(), 1)))
        assertEquals(UsbCardImportController.Progress.Started(3), run.progress.first())
        assertEquals(3, run.progress.count { it is UsbCardImportController.Progress.Each })
        assertEquals(UsbCardImportController.Progress.Done(setOf(1L, 3L), setOf(2L)), run.progress.last())
        assertTrue(run.failed.isEmpty()) // a persist refusal is a tally, not an onFailed
        assertEquals(1, camera.closes)
    }

    @Test fun `import refuses a photo whose card or metadata changed before any bytes move`() {
        val camera = Camera(mapOf(1L to jpeg("A.JPG"), 2L to jpeg("B.JPG", size = 200)))
        val otherCard = cardPhoto(camera, 1L).copy(cameraKey = "cam-b")
        val staleMetadata = CardPhoto(2L, "B.JPG", 100, cameraKey = "cam-a", objectInfo = jpeg("B.JPG", size = 100))
        val run = import(camera, listOf(otherCard, staleMetadata))

        assertTrue(camera.reads.isEmpty())
        assertEquals(setOf(1L, 2L), run.failed.keys)
        assertTrue(run.failed.getValue(1L), run.failed.getValue(1L).contains("card changed"))
        assertEquals(UsbCardImportController.Progress.Done(emptySet(), setOf(1L, 2L)), run.progress.last())
    }

    @Test fun `import stops at a full phone without touching the remaining selection`() {
        val camera = Camera(mapOf(1L to jpeg("A.JPG"), 2L to jpeg("B.JPG")))
        val photos = listOf(1L, 2L).map { cardPhoto(camera, it) }
        val full = LowCameraStorage()
        val run = import(camera, photos, persist = { _, _ -> throw full })

        assertEquals(listOf(1L), camera.reads)
        assertEquals(UsbCardImportController.Progress.Failed(full.message.orEmpty()), run.progress.last())
        assertTrue(run.progress.none { it is UsbCardImportController.Progress.Done })
        assertEquals(1, camera.closes)
    }

    @Test fun `import writes off the rest of the run once the session dies, and needs a device at all`() {
        val camera = object : Camera(mapOf(1L to jpeg("A.JPG"), 2L to jpeg("B.JPG"), 3L to jpeg("C.JPG"))) {
            override fun readObject(handle: Long, sink: OutputStream, expectedSize: Long?): Long {
                if (handle == 2L) { session.close(); throw PtpException("USB unplugged") }
                return super.readObject(handle, sink, expectedSize)
            }
        }
        val photos = listOf(1L, 2L, 3L).map { cardPhoto(camera, it) }
        val run = import(camera, photos)
        assertEquals(listOf(1L), camera.reads)
        assertEquals(UsbCardImportController.Progress.Done(setOf(1L), setOf(2L, 3L)), run.progress.last())
        assertEquals("USB unplugged", run.failed[2L])

        val idle = Camera(emptyMap())
        val noDevice = import(idle, photos, deviceProvider = { null })
        assertTrue((noDevice.progress.single() as UsbCardImportController.Progress.Failed).message.contains("no longer detected"))
        assertEquals(0, idle.closes)
    }
}
