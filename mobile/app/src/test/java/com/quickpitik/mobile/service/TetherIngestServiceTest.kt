package com.quickpitik.mobile.service

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.TetherEvents
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.usb.CameraIngestCoordinator
import com.quickpitik.mobile.support.closeRoom
import com.quickpitik.mobile.support.forgetRoomSingleton
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf

/**
 * The tether lifeline's two contracts that need no camera: the notification's
 * Stop action hands the decision to the ViewModel through TetherEvents and
 * lets the service end when nothing is running, and a start with nothing to
 * do promotes to foreground exactly once and then ends itself.
 */
@RunWith(RobolectricTestRunner::class)
class TetherIngestServiceTest {
    private val context: Context = ApplicationProvider.getApplicationContext()
    private lateinit var db: AppDatabase

    @Before fun setUp() {
        forgetRoomSingleton()
        context.deleteDatabase(AppDatabase.DB_NAME)
        db = AppDatabase.getDatabase(context)
        SessionManager.getInstance(context).apply {
            clearSession()
            saveSession("t", "PHOTOGRAPHER", "Juan", "shooter@example.com")
        }
        CameraIngestCoordinator::class.java.getDeclaredField("instance").apply {
            isAccessible = true
            set(null, null)
        }
    }

    @After fun tearDown() {
        closeRoom(context as android.app.Application)
    }

    @Test fun `a service with queued uploads and no camera stays in the foreground`() = runBlocking {
        db.uploadQueueDao().insertRecord(UploadRecord(filePath = "/cache/frame.jpg", eventId = "event",
            photographerId = "shooter@example.com", captureTimestamp = 1, uploadStatus = "QUEUED"))
        val update = Intent(context, TetherIngestService::class.java).setAction(TetherIngestService.ACTION_UPDATE)
        val controller = Robolectric.buildService(TetherIngestService::class.java, update).create()
        try {
            controller.startCommand(0, 1)
            val shadow = shadowOf(controller.get())
            assertFalse("Queued originals must keep the service alive before the first Room emission", shadow.isStoppedBySelf)
            val notices = shadowOf(context.getSystemService(android.app.NotificationManager::class.java))
            awaitMain { notices.allNotifications.any { it.extras.getString(android.app.Notification.EXTRA_TITLE) == "1 photos waiting to upload" } }
            assertNotNull(shadow.lastForegroundNotification)
            assertFalse(shadow.isStoppedBySelf)
            controller.get().onStartCommand(Intent(context, TetherIngestService::class.java)
                .setAction(TetherIngestService.ACTION_STOP), 0, 2)
            assertFalse("Stopping the camera must not stop queued uploads", shadow.isStoppedBySelf)
        } finally { controller.destroy() }
    }

    @Test fun `the service ends once the queue drains`() = runBlocking {
        val dao = db.uploadQueueDao()
        val id = dao.insertRecord(UploadRecord(filePath = "/cache/frame.jpg", eventId = "event",
            photographerId = "shooter@example.com", captureTimestamp = 1, uploadStatus = "UPLOADING"))
        dao.insertRecord(UploadRecord(filePath = "/cache/failed.jpg", eventId = "event",
            photographerId = "shooter@example.com", captureTimestamp = 1, uploadStatus = "FAILED"))
        dao.insertRecord(UploadRecord(filePath = "/cache/other.jpg", eventId = "event",
            photographerId = "other@example.com", captureTimestamp = 1, uploadStatus = "QUEUED"))
        val update = Intent(context, TetherIngestService::class.java).setAction(TetherIngestService.ACTION_UPDATE)
        val controller = Robolectric.buildService(TetherIngestService::class.java, update).create()
        try {
            controller.startCommand(0, 1)
            val shadow = shadowOf(controller.get())
            val notices = shadowOf(context.getSystemService(android.app.NotificationManager::class.java))
            awaitMain { notices.allNotifications.any { it.extras.getString(android.app.Notification.EXTRA_TITLE) == "1 photos waiting to upload" } }
            assertFalse(shadow.isStoppedBySelf)
            dao.updateStatus(id, "COMPLETED", null)
            awaitMain { shadow.isStoppedBySelf }
            assertTrue(shadow.isStoppedBySelf)
        } finally { controller.destroy() }
    }

    private fun awaitMain(ready: () -> Boolean) {
        val deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(5)
        while (!ready() && System.nanoTime() < deadline) {
            shadowOf(android.os.Looper.getMainLooper()).idle()
            Thread.sleep(10)
        }
        assertTrue("Service state did not settle", ready())
    }

    @Test fun `stop from the notification raises the stop signal and ends an idle service`() {
        val raised = CompletableDeferred<Unit>()
        val collector = CoroutineScope(Dispatchers.Unconfined).launch { raised.complete(TetherEvents.stopRequested.first()) }
        val stop = Intent(context, TetherIngestService::class.java).setAction(TetherIngestService.ACTION_STOP)
        val controller = Robolectric.buildService(TetherIngestService::class.java, stop).create()

        controller.startCommand(0, 1)

        assertTrue(raised.isCompleted)
        awaitMain { shadowOf(controller.get()).isStoppedBySelf }
        assertTrue(shadowOf(controller.get()).isStoppedBySelf)
        collector.cancel()
        controller.destroy()
    }

    @Test fun `a start with nothing running promotes to foreground and then ends itself`() {
        val update = Intent(context, TetherIngestService::class.java)
            .setAction(TetherIngestService.ACTION_UPDATE)
            .putExtra(TetherIngestService.EXTRA_NOTICE, "Camera transfer in progress")
        val controller = Robolectric.buildService(TetherIngestService::class.java, update).create()

        controller.startCommand(0, 1)

        val shadow = shadowOf(controller.get())
        assertNotNull(shadow.lastForegroundNotification)
        awaitMain { shadow.isStoppedBySelf }
        assertTrue(shadow.isStoppedBySelf)
        controller.destroy()
    }
}
