package com.quickpitik.mobile.ui.auth

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import okhttp3.mockwebserver.MockResponse
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * The Google exchange's two-step contract against a [MockWebServer] (the
 * RetrofitClient.setBaseUrl seam, same as PhotoUploadWorkerTest): a 422
 * ROLE_REQUIRED parks the ID token and asks for a role instead of erroring,
 * the completing call re-POSTs that token with the picked role and lands the
 * session in SessionManager, and cancelling drops the parked token so a late
 * "continue" can't replay it. The Credential Manager half needs a real device
 * with Play services — this covers everything after the ID token exists.
 */
@RunWith(RobolectricTestRunner::class)
class AuthViewModelGoogleTest {

    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private lateinit var session: SessionManager

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        session = SessionManager.getInstance(application)
        session.clearSession()
    }

    @After
    fun tearDown() {
        session.clearSession()
    }

    private fun roleRequired422() = MockResponse()
        .setResponseCode(422)
        .setBody(
            """{"success":false,"errors":[{"code":"ROLE_REQUIRED","message":"Choose a role to finish creating your account."}]}"""
        )

    private fun authSuccess200() = MockResponse()
        .setResponseCode(200)
        .setBody(
            """{"success":true,"data":{"accessToken":"access-1","refreshToken":"refresh-1","user":{"id":"u1","email":"juan@gmail.com","name":"Juan","role":"RUNNER","createdAt":"2026-08-29T00:00:00Z"}}}"""
        )

    @Test
    fun googleAdminMfaChallengeDoesNotCreateAnAndroidSession() {
        backend.enqueue(MockResponse().setResponseCode(200).setBody(
            """{"success":true,"data":{"accessToken":"","refreshToken":"","mfaRequired":true,"mfaToken":"challenge","user":{"id":"a1","email":"admin@test.local","name":"Admin","role":"ADMIN","createdAt":"2026-09-09T00:00:00Z"}}}"""
        ))
        val viewModel = AuthViewModel(application)
        viewModel.googleLogin("google-admin-token")
        val state = viewModel.authState.awaitFirst { it is AuthState.Error || it is AuthState.Success }
        assertTrue(state is AuthState.Error)
        org.junit.Assert.assertNull(session.getAccessToken())
        assertTrue((state as AuthState.Error).message.contains("website"))
    }

    @Test
    fun roleRequiredParksTheTokenAndAsksForARole() {
        val server = backend.enqueue(roleRequired422())
        val viewModel = AuthViewModel(application)

        viewModel.googleLogin("google-id-token")

        val state = viewModel.authState.awaitFirst {
            it is AuthState.GoogleRoleRequired || it is AuthState.Error
        }
        assertTrue("expected GoogleRoleRequired, got $state", state is AuthState.GoogleRoleRequired)
        assertEquals("/api/v1/auth/google", server.takeRequest().path)
    }

    @Test
    fun completingSignupRepostsTheParkedTokenWithTheRole() {
        val server = backend.enqueue(roleRequired422(), authSuccess200())
        val viewModel = AuthViewModel(application)

        viewModel.googleLogin("google-id-token")
        viewModel.authState.awaitFirst { it is AuthState.GoogleRoleRequired }

        viewModel.completeGoogleSignup(isPhotographer = false)

        val state = viewModel.authState.awaitFirst {
            it is AuthState.Success || it is AuthState.Error
        }
        assertTrue("expected Success, got $state", state is AuthState.Success)
        // The session landed exactly like a password login's would.
        assertEquals("access-1", session.getAccessToken())

        server.takeRequest() // the first (role-less) exchange
        val completing = server.takeRequest().body.readUtf8()
        assertTrue(completing.contains("\"idToken\":\"google-id-token\""))
        assertTrue(completing.contains("\"role\":\"RUNNER\""))
    }

    @Test
    fun cancellingDropsTheParkedTokenSoContinueCannotReplayIt() {
        backend.enqueue(roleRequired422())
        val viewModel = AuthViewModel(application)

        viewModel.googleLogin("google-id-token")
        viewModel.authState.awaitFirst { it is AuthState.GoogleRoleRequired }

        viewModel.cancelGoogleSignup()
        // No response is enqueued past the 422 — this must fail locally,
        // before any network call, or the test times out on a hung request.
        viewModel.completeGoogleSignup(isPhotographer = true)

        val state = viewModel.authState.awaitFirst { it is AuthState.Error }
        assertTrue((state as AuthState.Error).message.contains("expired"))
    }
}
