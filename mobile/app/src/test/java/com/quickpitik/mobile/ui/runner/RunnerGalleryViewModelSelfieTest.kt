package com.quickpitik.mobile.ui.runner

import android.app.Application
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.withTimeout
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
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
@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
class RunnerGalleryViewModelSelfieTest {

    private lateinit var application: Application
    private lateinit var session: SessionManager
    private var server: MockWebServer? = null

    @Before
    fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        application = ApplicationProvider.getApplicationContext()
        session = SessionManager.getInstance(application)
        session.clearSession()
        session.saveSession("t", "RUNNER", "Juan", "juan@x.com")
    }

    @After
    fun tearDown() {
        server?.shutdown()
        server = null
        RetrofitClient.resetBaseUrl()
        session.clearSession()
        Dispatchers.resetMain()
    }

    private fun startServer(vararg responses: MockResponse): MockWebServer =
        MockWebServer().apply {
            responses.forEach { enqueue(it) }
            start()
            server = this
            RetrofitClient.setBaseUrl(url("/").toString())
        }

    private fun ok(data: String) = MockResponse().setBody("""{"success":true,"data":$data}""")

    private val selfieJson =
        """{"id":"s1","dataUrl":"","uploadedAt":"","isPrimary":true,"qualityScore":0,"qualityTestStatus":"untested"}"""

    private fun awaitAlert(
        viewModel: RunnerGalleryViewModel,
        accept: (PhotoAlertUiState) -> Boolean,
    ): PhotoAlertUiState = runBlocking {
        withTimeout(5_000) { viewModel.photoAlert.first(accept) }
    }

    private fun pickedSelfie(): Uri {
        val uri = Uri.parse("content://test/selfie.jpg")
        Shadows.shadowOf(application.contentResolver)
            .registerInputStream(uri, ByteArrayInputStream(ByteArray(16)))
        return uri
    }

    @Test
    fun addingASelfieFlipsTheCardToReadyWithoutRegistering() {
        val server = startServer(
            ok("[]"),                                    // GET selfies → NeedsSelfie
            ok(selfieJson),                              // POST selfies
            ok("[$selfieJson]"),                         // GET selfies (refresh)
            ok("""{"registered":false,"selfieId":null}"""), // GET photo-alert
        )
        val viewModel = RunnerGalleryViewModel(application)
        viewModel.loadGalleryMetadata("ev")
        awaitAlert(viewModel) { it is PhotoAlertUiState.NeedsSelfie }

        viewModel.addSelfieToLibrary(pickedSelfie())

        val state = awaitAlert(viewModel) {
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
        startServer(
            ok("[]"),
            MockResponse().setResponseCode(422).setBody(
                """{"success":false,"errors":[{"code":"SELFIE_REJECTED","message":"No face found"}]}"""
            ),
        )
        val viewModel = RunnerGalleryViewModel(application)
        viewModel.loadGalleryMetadata("ev")
        awaitAlert(viewModel) { it is PhotoAlertUiState.NeedsSelfie }

        viewModel.addSelfieToLibrary(pickedSelfie())

        val state = awaitAlert(viewModel) {
            it is PhotoAlertUiState.Ready || (it as? PhotoAlertUiState.NeedsSelfie)?.message != null
        }
        assertTrue("expected NeedsSelfie, got $state", state is PhotoAlertUiState.NeedsSelfie)
        state as PhotoAlertUiState.NeedsSelfie
        assertNotNull(state.message)
        assertEquals(false, state.uploading)
    }
}
