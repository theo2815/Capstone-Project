package com.quickpitik.mobile.service

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.TetherEvents
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
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

    @Test fun `stop from the notification raises the stop signal and ends an idle service`() {
        val raised = CompletableDeferred<Unit>()
        val collector = CoroutineScope(Dispatchers.Unconfined).launch { raised.complete(TetherEvents.stopRequested.first()) }
        val stop = Intent(context, TetherIngestService::class.java).setAction(TetherIngestService.ACTION_STOP)
        val controller = Robolectric.buildService(TetherIngestService::class.java, stop).create()

        controller.startCommand(0, 1)

        assertTrue(raised.isCompleted)
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
        assertTrue(shadow.isStoppedBySelf)
        controller.destroy()
    }
}
