package com.quickpitik.mobile.data.usb.ptp

import android.content.Context
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.*
import kotlinx.coroutines.*
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadow.api.Shadow
import org.robolectric.shadows.ShadowStatFs
import java.io.OutputStream

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
class UsbEventCaptureControllerTest {
    private val info = PtpDeviceInfo(100, 0, 0, "",
        listOf(Ptp.OP_GET_OBJECT_HANDLES, Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT),
        emptyList(), emptyList(), emptyList(), listOf(0x3801), "Test", "Camera", "1", "serial")

    @Before fun clearSpool() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        ShadowStatFs.registerStats(context.filesDir.absolutePath, 4_000_000, 2_000_000, 2_000_000)
        UploadSpool.dir(ApplicationProvider.getApplicationContext()).listFiles()
            ?.filter { it.name.startsWith("camera_") }?.forEach { it.delete() }
    }

    private open inner class Camera(val list: () -> List<Long>) : GenericPtpCamera(PtpSession(ScriptedPtp()), info, "camera") {
        var metadataReads = 0
        var closes = 0
        val transfers = mutableListOf<Long>()
        override fun handles() = list()
        override fun objectInfo(handle: Long): PtpObjectInfo {
            metadataReads++
            return PtpObjectInfo(0x3801, 4, "same-name.jpg", 1, 1, "shot-$handle")
        }
        override fun readObject(handle: Long, sink: OutputStream, expectedSize: Long?): Long {
            transfers.add(handle)
            sink.write(byteArrayOf(0xff.toByte(), 0xd8.toByte(), handle.toByte(), (handle shr 8).toByte()))
            return 4
        }
        override fun close() { closes++ }
    }

    private suspend fun watch(controller: UsbEventCaptureController, capture: (String) -> Unit = {}) = controller.run(
        "owner", "event", { true }, {}, {}, {}, {}, capture,
    )

    @Test fun `camera events that never announce new photos are disabled after ten card-detected shots`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            val store = CameraTransferStore(context, db)
            var scans = 0
            var eventCalls = 0
            val camera = object : Camera({ scans++; if (scans == 1) listOf(1) else (1L..scans.toLong()).toList() }) {
                override val capabilities get() = super.capabilities.copy(eventPolling = true)
                override fun events(): List<Long> { eventCalls++; return emptyList() }
            }
            val device = Shadow.newInstanceOf(UsbDevice::class.java)
            val controller = UsbEventCaptureController(context.getSystemService(Context.USB_SERVICE) as UsbManager,
                { device }, store, openCamera = { _, _ -> camera }, nowMs = { testScheduler.currentTime })
            var captured = 0
            try { watch(controller) { if (++captured == 14) throw CancellationException("Shoot finished") } }
            catch (_: CancellationException) { }
            assertTrue("scans=$scans", scans >= 14)
            assertEquals(11, eventCalls)
        } finally { db.close() }
    }

    @Test fun `live watch pauses before reading a full phone and resumes the same photo after cleanup`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            val store = CameraTransferStore(context, db)
            var scans = 0
            val camera = Camera {
                if (scans++ == 0) listOf(1) else {
                    if (scans == 2) ShadowStatFs.registerStats(context.filesDir.absolutePath, 4_000_000, 0, 0)
                    listOf(1, 2)
                }
            }
            val device = Shadow.newInstanceOf(UsbDevice::class.java)
            val controller = UsbEventCaptureController(context.getSystemService(Context.USB_SERVICE) as UsbManager,
                { device }, store, openCamera = { _, _ -> camera }, nowMs = { testScheduler.currentTime })
            var paused = false
            try {
                controller.run("owner", "event", { true }, {}, {}, {}, { reason ->
                    if (reason != null) {
                        paused = true
                        assertTrue(camera.transfers.isEmpty())
                        ShadowStatFs.registerStats(context.filesDir.absolutePath, 4_000_000, 2_000_000, 2_000_000)
                    }
                }, { throw CancellationException("Photo safely queued") })
            } catch (_: CancellationException) { }
            assertTrue(paused)
            assertEquals(listOf(2L), camera.transfers)
            assertEquals(1, db.uploadQueueDao().getRecordsWithStatus("QUEUED").size)
            assertTrue(store.dao.forScope(fingerprint("owner", "event", "camera")).all { it.attempts == 0 })
        } finally { db.close() }
    }

    @Test fun `one hundred rapid shots preserve every original and ignore the baseline`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            val store = CameraTransferStore(context, db)
            var scans = 0
            val camera = Camera { if (scans++ == 0) listOf(1) else (1L..101L).toList() }
            val device = Shadow.newInstanceOf(UsbDevice::class.java)
            val controller = UsbEventCaptureController(context.getSystemService(Context.USB_SERVICE) as UsbManager,
                { device }, store, openCamera = { _, _ -> camera }, nowMs = { testScheduler.currentTime })
            var captured = 0
            try { watch(controller) { if (++captured == 100) throw CancellationException("Shoot finished") } }
            catch (_: CancellationException) { }
            assertEquals((2L..101L).toList(), camera.transfers)
            assertEquals(100, db.uploadQueueDao().getRecordsWithStatus("QUEUED").size)
            assertEquals(1, camera.closes)
            assertFalse(UploadSpool.dir(context).listFiles().orEmpty().any { it.name.endsWith(".part") })
        } finally { db.close() }
    }

    @Test fun `reconnect catches up missed photos without reimporting completed transfers`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            var scans = 0
            val first = Camera { when (++scans) { 1 -> listOf(1); 2 -> listOf(1, 2); else -> throw PtpException("USB unplugged") } }
            val second = Camera { (1L..4L).toList() }
            var opens = 0
            val device = Shadow.newInstanceOf(UsbDevice::class.java)
            val controller = UsbEventCaptureController(context.getSystemService(Context.USB_SERVICE) as UsbManager,
                { device }, CameraTransferStore(context, db), openCamera = { _, _ -> if (opens++ == 0) first else second },
                nowMs = { testScheduler.currentTime })
            var captured = 0
            try { watch(controller) { if (++captured == 3) throw CancellationException("Shoot finished") } }
            catch (_: CancellationException) { }
            assertEquals(listOf(2L), first.transfers)
            assertEquals(listOf(3L, 4L), second.transfers)
            assertEquals(3, db.uploadQueueDao().getRecordsWithStatus("QUEUED").size)
            assertEquals(1, first.closes)
            assertEquals(1, second.closes)
        } finally { db.close() }
    }

    @Test fun `flapping sessions exhaust recovery with bounded handshakes`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            var opens = 0
            val device = Shadow.newInstanceOf(UsbDevice::class.java)
            val controller = UsbEventCaptureController(context.getSystemService(Context.USB_SERVICE) as UsbManager,
                { device }, CameraTransferStore(context, db), openCamera = { _, _ ->
                    opens++
                    var scans = 0
                    Camera { if (scans++ == 0) emptyList() else throw PtpException("Link lost") }
                }, nowMs = { testScheduler.currentTime })
            try { watch(controller); fail("Expected a bounded recovery failure") }
            catch (e: PtpException) { assertTrue(e.message.orEmpty().contains("timed out")) }
            assertTrue("Flapping must not keep extending the recovery budget", testScheduler.currentTime < 330_000)
            assertTrue("Backoff must prevent a handshake storm: $opens", opens <= 16)
        } finally { db.close() }
    }

    @Test fun `idle monitoring does not reread unchanged metadata every scan`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            val camera = Camera {
                if (testScheduler.currentTime >= 60_000) throw CancellationException("Idle observation finished")
                (1L..100L).toList()
            }
            val device = Shadow.newInstanceOf(UsbDevice::class.java)
            val controller = UsbEventCaptureController(context.getSystemService(Context.USB_SERVICE) as UsbManager,
                { device }, CameraTransferStore(context, db), openCamera = { _, _ -> camera }, nowMs = { testScheduler.currentTime })
            try { watch(controller) } catch (_: CancellationException) { }
            assertEquals(108, camera.metadataReads)
            assertTrue(camera.transfers.isEmpty())
            assertEquals(1, camera.closes)
        } finally { db.close() }
    }

    @Test fun `a reused handle is reidentified before transfer and never attributed to the old shot`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            var scans = 0
            var reads = 0
            val camera = object : Camera({ if (scans++ == 0) listOf(1) else listOf(1, 2) }) {
                override fun objectInfo(handle: Long): PtpObjectInfo {
                    val original = super.objectInfo(handle)
                    return if (handle == 2L && reads++ > 0) original.copy(capturedAt = "replacement") else original
                }
            }
            val device = Shadow.newInstanceOf(UsbDevice::class.java)
            val store = CameraTransferStore(context, db)
            val controller = UsbEventCaptureController(context.getSystemService(Context.USB_SERVICE) as UsbManager,
                { device }, store, openCamera = { _, _ -> camera }, nowMs = { testScheduler.currentTime })
            try { watch(controller) { throw CancellationException("Replacement captured") } } catch (_: CancellationException) { }
            assertEquals(listOf(2L), camera.transfers)
            val ledger = store.dao.forScope(fingerprint("owner", "event", "camera"))
            assertEquals("FAILED", ledger.single { it.handle == 2L && it.status != "QUEUED" }.status)
            assertEquals(1, ledger.count { it.status == "QUEUED" })
        } finally { db.close() }
    }

    @Test fun `exhausted photo waits for manual retry without repeated transfers`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            val store = CameraTransferStore(context, db)
            val record = store.transfer("owner", "event", "camera", 2,
                PtpObjectInfo(0x3801, 4, "same-name.jpg", 1, 1, "shot-2"))
            store.reconcile(record.scope, emptyList())
            store.dao.insert(listOf(record.copy(status = "FAILED", attempts = CameraTransferStore.MAX_TRANSFER_ATTEMPTS)))
            var scans = 0
            val camera = Camera { if (++scans > 3) throw CancellationException("Retry budget observed") else listOf(2) }
            val device = Shadow.newInstanceOf(UsbDevice::class.java)
            val controller = UsbEventCaptureController(context.getSystemService(Context.USB_SERVICE) as UsbManager,
                { device }, store, openCamera = { _, _ -> camera }, nowMs = { testScheduler.currentTime })
            try { watch(controller) } catch (_: CancellationException) { }
            assertTrue(camera.transfers.isEmpty())
            assertEquals(5, store.dao.get(record.key)!!.attempts)
            store.dao.retry("owner", "event")
            assertEquals(0, store.dao.get(record.key)!!.attempts)
            assertEquals("PENDING", store.dao.get(record.key)!!.status)
        } finally { db.close() }
    }
}
