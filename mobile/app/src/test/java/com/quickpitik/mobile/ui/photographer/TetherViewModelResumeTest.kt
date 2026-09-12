package com.quickpitik.mobile.ui.photographer

import android.app.Application
import android.content.Context
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.closeRoom
import com.quickpitik.mobile.support.signInAs
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Resuming a shutter watch after process death: the persisted snapshot is
 * restored into the active event only for the account that started it and
 * only when it names the event the watch was on. A bug here resumes a shoot
 * into the wrong account or the wrong event.
 *
 * Moved from PhotographerDashboardViewModelResumeTest with the ViewModel split
 * (plan 2 phase G). The second resume path - looking the event up in the
 * dashboard's list when the snapshot is unusable - is still uncovered: it only
 * runs once the camera reports Connected, which needs the USB shadow
 * machinery CameraConnectionManagerTest builds. The events flow is a
 * constructor parameter now, so that test no longer needs a second ViewModel.
 */
@RunWith(RobolectricTestRunner::class)
class TetherViewModelResumeTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val viewModels = ViewModelStore()
    private val tether get() = application.getSharedPreferences("quickpitik_tether", Context.MODE_PRIVATE)

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        tether.edit().clear().commit()
    }

    @After fun tearDown() {
        viewModels.clear()
        // These tests finish within milliseconds of constructing the ViewModel,
        // so its startup Room queries are still on the executor.
        closeRoom(application)
        tether.edit().clear().commit()
        SessionManager.getInstance(application).clearSession()
    }

    private fun persistWatch(eventId: String, owner: String, snapshotId: String) {
        tether.edit()
            .putString("watch_active_event", eventId)
            .putString("watch_owner", owner)
            .putString("watch_event_snapshot", """{"id":"$snapshotId","slug":"cebu","name":"Cebu Run","date":"2026-09-11","location":"Cebu","state":"live","photoCount":0,"salesCount":0,"revenueKept":0.0,"bannerUrl":null,"handle":null}""")
            .commit()
    }

    private fun model() = TetherViewModel(application).also { viewModels.put("tether", it) }

    @Test fun `the saved event comes back for the account that started the watch`() {
        persistWatch(eventId = "ev-1", owner = "juan@x.com", snapshotId = "ev-1")
        assertEquals("ev-1", model().activeEvent.value?.id)
    }

    @Test fun `another account's watch, or a snapshot for a different event, is never restored`() {
        persistWatch(eventId = "ev-1", owner = "someone-else@x.com", snapshotId = "ev-1")
        assertNull(model().activeEvent.value)

        persistWatch(eventId = "ev-2", owner = "juan@x.com", snapshotId = "ev-1")
        assertNull(model().activeEvent.value)
    }

    @Test fun `no persisted watch leaves the tab on its event picker`() {
        assertNull(model().activeEvent.value)
    }
}
