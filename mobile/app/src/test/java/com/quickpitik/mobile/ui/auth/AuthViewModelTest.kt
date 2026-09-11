package com.quickpitik.mobile.ui.auth

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.ViewMode
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.TimeUnit

/**
 * Password login, registration, the OTP reset chain and sign-out — everything
 * in AuthViewModel except the Google leg, which AuthViewModelGoogleTest covers.
 * The one-shot reset token and the parked Google token share the same rule:
 * memory only, consumed once, dropped by a fresh request.
 */
@RunWith(RobolectricTestRunner::class)
class AuthViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private lateinit var session: SessionManager
    private val seen = CopyOnWriteArrayList<Pair<String, String>>() // path to body
    @Volatile private var respond: (path: String, body: String) -> MockResponse = { _, _ -> MockResponse().setResponseCode(404) }

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        session = SessionManager.getInstance(application).also { it.clearSession() }
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val path = request.path.orEmpty()
                val body = request.body.readUtf8()
                seen += path to body
                return respond(path, body)
            }
        }
    }

    @After fun tearDown() = session.clearSession()

    private fun bodies(path: String) = seen.filter { it.first == path }.map { it.second }
    private fun AuthViewModel.awaitAuth() = authState.awaitFirst { it is AuthState.Success || it is AuthState.Error }
    private fun AuthViewModel.awaitReset() =
        passwordResetState.awaitFirst { it is PasswordResetState.Success || it is PasswordResetState.Error }

    @Test fun `login lands a complete session, refuses a blank one, and shows the backend's own message`() {
        respond = { _, _ -> ok(AUTH) }
        val vm = AuthViewModel(application)
        vm.login("juan@x.com", "secret123")
        assertTrue(vm.awaitAuth() is AuthState.Success)
        assertEquals("access-1", session.getAccessToken())
        assertEquals("refresh-1", session.getRefreshToken())
        assertEquals("RUNNER", session.getUserRole())
        assertEquals("juan@x.com", session.getUserEmail())
        session.clearSession()

        respond = { _, _ -> ok(BLANK_AUTH) }
        vm.login("juan@x.com", "secret123")
        val blank = vm.awaitAuth() as AuthState.Error
        assertTrue(blank.message, blank.message.contains("did not return a session"))
        assertNull(session.getAccessToken())

        respond = { _, _ ->
            MockResponse().setResponseCode(401).setBody(
                """{"success":false,"errors":[{"code":"INVALID_CREDENTIALS","message":"Wrong email or password."}]}""",
            )
        }
        vm.login("juan@x.com", "wrong")
        assertEquals("Wrong email or password.", (vm.awaitAuth() as AuthState.Error).message)
        assertNull(session.getAccessToken())
    }

    @Test fun `register sends the picked role and lands the session`() {
        respond = { _, _ -> ok(AUTH_PHOTOGRAPHER) }
        val vm = AuthViewModel(application)
        vm.register("Juan", "juan@x.com", "secret123", isPhotographer = true)
        assertTrue(vm.awaitAuth() is AuthState.Success)
        assertTrue(bodies("/api/v1/auth/register").single().contains(""""role":"PHOTOGRAPHER""""))
        assertEquals("PHOTOGRAPHER", session.getUserRole())

        vm.register("Ana", "ana@x.com", "secret123", isPhotographer = false)
        vm.awaitAuth()
        assertTrue(bodies("/api/v1/auth/register").last().contains(""""role":"RUNNER""""))
    }

    @Test fun `the reset chain parks a one-shot token on verify, consumes it on confirm, drops it on a new request`() {
        respond = { path, body ->
            when (path) {
                "/api/v1/auth/forgot-password" -> ok("""{"message":"If that address exists, a code is on its way."}""")
                "/api/v1/auth/verify-reset-otp" ->
                    if (body.contains("999999")) {
                        MockResponse().setResponseCode(422).setBody(
                            """{"success":false,"errors":[{"code":"RESET_CODE_INVALID","message":"Wrong code."}]}""",
                        )
                    } else ok("""{"resetToken":"rt-1"}""")
                "/api/v1/auth/reset-password" -> ok("""{"message":"Password updated."}""")
                else -> MockResponse().setResponseCode(404)
            }
        }
        val vm = AuthViewModel(application)

        // Nothing parked yet: fails locally, never reaches the server.
        vm.confirmPasswordReset("newpass123")
        assertTrue((vm.awaitReset() as PasswordResetState.Error).message.contains("expired"))
        assertTrue(bodies("/api/v1/auth/reset-password").isEmpty())

        vm.requestPasswordReset("  juan@x.com ")
        assertEquals(PasswordResetState.Success, vm.awaitReset())
        assertTrue(bodies("/api/v1/auth/forgot-password").single().contains(""""email":"juan@x.com""""))

        vm.verifyResetOtp("juan@x.com", "123456")
        assertEquals(PasswordResetState.Success, vm.awaitReset())
        vm.confirmPasswordReset("newpass123")
        assertEquals(PasswordResetState.Success, vm.awaitReset())
        val reset = bodies("/api/v1/auth/reset-password").single()
        assertTrue(reset, reset.contains(""""token":"rt-1"""") && reset.contains(""""newPassword":"newpass123""""))

        // Consumed: a second confirm has nothing to send.
        vm.confirmPasswordReset("newpass123")
        assertTrue(vm.awaitReset() is PasswordResetState.Error)
        assertEquals(1, bodies("/api/v1/auth/reset-password").size)

        // A fresh request invalidates a parked token.
        vm.verifyResetOtp("juan@x.com", "123456")
        vm.awaitReset()
        vm.requestPasswordReset("juan@x.com")
        vm.awaitReset()
        vm.confirmPasswordReset("newpass123")
        assertTrue(vm.awaitReset() is PasswordResetState.Error)
        assertEquals(1, bodies("/api/v1/auth/reset-password").size)

        // A refused code surfaces the backend's wording and parks nothing.
        vm.verifyResetOtp("juan@x.com", "999999")
        assertEquals("Wrong code.", (vm.awaitReset() as PasswordResetState.Error).message)
        vm.confirmPasswordReset("newpass123")
        assertTrue(vm.awaitReset() is PasswordResetState.Error)
        assertEquals(1, bodies("/api/v1/auth/reset-password").size)
    }

    @Test fun `logout clears the session first, resets the view, then revokes best-effort`() {
        respond = { _, _ -> MockResponse().setResponseCode(500) } // a failed revoke must not matter
        session.saveSession("access-1", "PHOTOGRAPHER", "Juan", "juan@x.com", refreshToken = "refresh-1")
        ViewMode.set(session, true)
        val vm = AuthViewModel(application)

        vm.logout()
        assertNull(session.getAccessToken())
        assertNull(session.getRefreshToken())
        assertFalse(ViewMode.runnerView.value)
        assertFalse(session.isRunnerView())
        assertEquals(AuthState.Idle, vm.authState.value)
        val revoke = requireNotNull(backend.server.takeRequest(5, TimeUnit.SECONDS))
        assertEquals("/api/v1/auth/logout", revoke.path)
        assertTrue(bodies("/api/v1/auth/logout").single().contains(""""refreshToken":"refresh-1""""))

        // Nothing to revoke: no request at all.
        session.saveSession("access-2", "RUNNER", "Ana", "ana@x.com")
        vm.logout()
        assertNull(session.getAccessToken())
        assertNull(backend.server.takeRequest(500, TimeUnit.MILLISECONDS))
    }

    private companion object {
        const val AUTH = """{"accessToken":"access-1","refreshToken":"refresh-1","user":{"id":"u1","email":"juan@x.com","name":"Juan","role":"RUNNER","createdAt":"2026-09-11T00:00:00Z"}}"""
        const val BLANK_AUTH = """{"accessToken":"","refreshToken":"refresh-1","user":{"id":"u1","email":"juan@x.com","name":"Juan","role":"RUNNER","createdAt":"2026-09-11T00:00:00Z"}}"""
        const val AUTH_PHOTOGRAPHER = """{"accessToken":"access-p","refreshToken":"refresh-p","user":{"id":"u2","email":"juan@x.com","name":"Juan","role":"PHOTOGRAPHER","createdAt":"2026-09-11T00:00:00Z"}}"""
    }
}
