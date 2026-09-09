package com.quickpitik.mobile.data.usb

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.*
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.test.resetMain
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
class CameraIngestCoordinatorTest {
    @Test fun `card operation waits for its own service start and rejects overlap`() = runTest {
        Dispatchers.setMain(StandardTestDispatcher(testScheduler))
        try {
            val context = ApplicationProvider.getApplicationContext<android.app.Application>()
            val coordinator = CameraIngestCoordinator.get(context)
            val entered = CompletableDeferred<Unit>()
            val job = launch { coordinator.runCard { entered.complete(Unit); awaitCancellation() } }
            runCurrent()
            val intent = requireNotNull(shadowOf(context).nextStartedService)
            val generation = intent.getLongExtra("card_generation", -1)
            coordinator.beginCard(this, generation - 1)
            runCurrent()
            assertFalse(entered.isCompleted)
            try { coordinator.runCard {}; fail("Expected exclusive camera ownership") }
            catch (_: IllegalStateException) { }
            coordinator.beginCard(this, generation)
            entered.await()
            job.cancelAndJoin()
            assertFalse(coordinator.cardBusy.value)
        } finally { Dispatchers.resetMain() }
    }

    @Test fun `stop remains busy until cancellation cleanup finishes`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val coordinator = CameraIngestCoordinator.get(context)
        val cleanup = CompletableDeferred<Unit>()
        val started = CompletableDeferred<Unit>()
        val job = launch {
            try { started.complete(Unit); awaitCancellation() }
            finally { withContext(NonCancellable) { cleanup.await() } }
        }
        started.await()
        CameraIngestCoordinator::class.java.getDeclaredField("watchJob").apply {
            isAccessible = true; set(coordinator, job)
        }
        coordinator.watch.value = ShutterWatchState.Watching(1, "photo.jpg", emptyList())
        coordinator.stopWatch()
        runCurrent()
        assertTrue(coordinator.watch.value is ShutterWatchState.Stopping)
        coordinator.requestWatch("owner", "event")
        assertTrue(coordinator.watch.value is ShutterWatchState.Stopping)
        cleanup.complete(Unit)
        job.join()
        assertTrue(coordinator.watch.value is ShutterWatchState.Idle)
    }
}
