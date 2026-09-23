package com.quickpitik.mobile.ui.runner

import android.app.Application
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.MockResponse
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import java.io.ByteArrayInputStream

/**
 * "Add a selfie →" on the Photo Alerts card adds to the library IN PLACE and
 * the card re-derives itself from the refreshed library — no Profile detour,
 * and no auto-register (website parity: the runner still taps "Notify me").
 * Same MockWebServer harness as [com.quickpitik.mobile.ui.auth.AuthViewModelGoogleTest].
 */
@RunWith(RobolectricTestRunner::class)
class RunnerGalleryViewModelSelfieTest {

    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private lateinit var session: SessionManager

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        session = signInAs("RUNNER")
    }

    @After
    fun tearDown() {
        session.clearSession()
    }

    private val selfieJson =
        """{"id":"s1","dataUrl":"","uploadedAt":"","isPrimary":true,"qualityScore":0,"qualityTestStatus":"untested"}"""

    private fun pickedSelfie(): Uri {
        val uri = Uri.parse("content://test/selfie.jpg")
        Shadows.shadowOf(application.contentResolver)
            .registerInputStream(uri, ByteArrayInputStream(ByteArray(16)))
        return uri
    }

    @Test
    fun addingASelfieFlipsTheCardToReadyWithoutRegistering() {
        val server = backend.enqueue(
            ok("[]"),                                    // GET selfies → NeedsSelfie
            ok(selfieJson),                              // POST selfies
            ok("[$selfieJson]"),                         // GET selfies (refresh)
            ok("""{"registered":false,"selfieId":null}"""), // GET photo-alert
        )
        val viewModel = RunnerGalleryViewModel(application)
        viewModel.loadGalleryMetadata("ev")
        viewModel.photoAlert.awaitFirst { it is PhotoAlertUiState.NeedsSelfie }

        viewModel.addSelfieToLibrary(pickedSelfie())

        val state = viewModel.photoAlert.awaitFirst {
            it is PhotoAlertUiState.Ready || (it as? PhotoAlertUiState.NeedsSelfie)?.message != null
        }
        assertEquals(PhotoAlertUiState.Ready(registered = false), state)

        val paths = (1..4).map { server.takeRequest().let { "${it.method} ${it.path}" } }
        assertEquals(
            listOf(
                "GET /api/v1/me/selfies",
                "POST /api/v1/me/selfies",
                "GET /api/v1/me/selfies",
                "GET /api/v1/events/ev/photo-alert",
            ),
            paths,
        )
        // No POST photo-alert: adding a selfie never opts the runner in by itself.
        assertEquals(4, server.requestCount)
    }

    @Test
    fun failedUploadStaysOnTheCardWithAMessage() {
        backend.enqueue(
            ok("[]"),
            MockResponse().setResponseCode(422).setBody(
                """{"success":false,"errors":[{"code":"SELFIE_REJECTED","message":"No face found"}]}"""
            ),
        )
        val viewModel = RunnerGalleryViewModel(application)
        viewModel.loadGalleryMetadata("ev")
        viewModel.photoAlert.awaitFirst { it is PhotoAlertUiState.NeedsSelfie }

        viewModel.addSelfieToLibrary(pickedSelfie())

        val state = viewModel.photoAlert.awaitFirst {
            it is PhotoAlertUiState.Ready || (it as? PhotoAlertUiState.NeedsSelfie)?.message != null
        }
        assertTrue("expected NeedsSelfie, got $state", state is PhotoAlertUiState.NeedsSelfie)
        state as PhotoAlertUiState.NeedsSelfie
        assertNotNull(state.message)
        assertEquals(false, state.uploading)
    }
}
