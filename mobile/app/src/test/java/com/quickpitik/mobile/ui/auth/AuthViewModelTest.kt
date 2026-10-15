package com.quickpitik.mobile.ui.auth

import android.app.Application
import android.content.Context
import android.content.ContextWrapper
import android.content.SharedPreferences
import android.os.Bundle
import android.os.Parcel
import androidx.lifecycle.SavedStateHandle
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.ViewMode
import com.quickpitik.mobile.data.remote.RetrofitClient
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
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.Serializable
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Password login, registration, the OTP reset chain, sign-out and the two
 * email links — everything in AuthViewModel except the Google leg, which
 * AuthViewModelGoogleTest covers. The one-shot reset token and the parked
 * Google token share the same rule: memory only, consumed once, dropped by a
 * fresh request. Email-link tokens are one-shot on the backend instead, which
 * fails a spent one exactly like a forged one: each is sent once.
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
    private fun AuthViewModel.awaitLink(token: String) = emailLinks.awaitFirst { token in it }.getValue(token)

    /**
     * What survives a process death: the handle's values, written to a Parcel
     * the way the system writes the saved-state Bundle, then read back.
     */
    private fun afterProcessDeath(handle: SavedStateHandle): SavedStateHandle {
        val out = Bundle().apply { handle.keys().forEach { putSerializable(it, handle.get<Serializable>(it)) } }
        val parcel = Parcel.obtain()
        return try {
            parcel.writeBundle(out)
            parcel.setDataPosition(0)
            val back = requireNotNull(parcel.readBundle(javaClass.classLoader))
            SavedStateHandle(back.keySet().associateWith { back.getSerializable(it, Serializable::class.java) })
        } finally {
            parcel.recycle()
        }
    }

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

        // A 200 the backend refuses inside the envelope. That is the other
        // branch: the 401 above is thrown and read by the catch, this one is
        // read off the envelope itself.
        respond = { _, _ ->
            MockResponse().setBody(
                """{"success":false,"errors":[{"code":"ACCOUNT_LOCKED","message":"Too many tries. Try again in 15 minutes."}]}""",
            )
        }
        vm.login("juan@x.com", "wrong")
        assertEquals("Too many tries. Try again in 15 minutes.", (vm.awaitAuth() as AuthState.Error).message)
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
        val recorded = CountDownLatch(1)
        respond = { _, _ -> recorded.countDown(); MockResponse().setResponseCode(500) } // a failed revoke must not matter
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
        // takeRequest returns before the dispatcher runs; wait for it to record the body.
        assertTrue(recorded.await(5, TimeUnit.SECONDS))
        assertTrue(bodies("/api/v1/auth/logout").single().contains(""""refreshToken":"refresh-1""""))

        // Nothing to revoke: no request at all.
        session.saveSession("access-2", "RUNNER", "Ana", "ana@x.com")
        vm.logout()
        assertNull(session.getAccessToken())
        assertNull(backend.server.takeRequest(500, TimeUnit.MILLISECONDS))
    }

    @Test fun `a verify link is sent once however often the screen asks, and a refusal reads as the backend's words`() {
        val release = CountDownLatch(1)
        respond = { _, _ -> release.await(5, TimeUnit.SECONDS); ok("""{"message":"Email confirmed. Thanks!"}""") }
        val vm = AuthViewModel(application)

        vm.verifyEmail("v-1")
        vm.verifyEmail("v-1") // a rotation while the POST is still out
        release.countDown()
        assertEquals(EmailLinkState.Success("Email confirmed. Thanks!"), vm.awaitLink("v-1"))
        vm.verifyEmail("v-1") // a rotation after the answer
        assertEquals("/api/v1/auth/verify-email", backend.server.takeRequest(5, TimeUnit.SECONDS)?.path)
        assertNull(backend.server.takeRequest(500, TimeUnit.MILLISECONDS))
        assertTrue(bodies("/api/v1/auth/verify-email").single().contains(""""token":"v-1""""))

        // A spent token fails like a forged one: the backend's words, not "HTTP 400".
        respond = { _, _ ->
            MockResponse().setResponseCode(400).setBody(
                """{"success":false,"errors":[{"code":"INVALID_VERIFICATION_TOKEN","message":"Verification link expired or already used"}]}""",
            )
        }
        vm.verifyEmail("v-2")
        assertEquals(EmailLinkState.Error("Verification link expired or already used"), vm.awaitLink("v-2"))

        // A link without a token fails on the phone.
        vm.verifyEmail("")
        assertEquals(EmailLinkState.Error("Invalid or missing verification token."), vm.awaitLink(""))
        assertEquals(2, bodies("/api/v1/auth/verify-email").size)
    }

    @Test fun `a confirmed email change signs this phone out and shows the backend's message, a refused one keeps the session`() {
        respond = { path, _ ->
            if (path == "/api/v1/auth/refresh") {
                MockResponse().setResponseCode(401).setBody(
                    """{"success":false,"errors":[{"code":"INVALID_REFRESH_TOKEN","message":"Refresh token expired or revoked"}]}""",
                )
            } else {
                ok(CONFIRMED)
            }
        }
        session.saveSession("access-1", "RUNNER", "Juan", "juan@x.com", refreshToken = "refresh-1")
        val vm = AuthViewModel(application)

        vm.confirmEmailChange("c-1")
        assertEquals(EmailLinkState.Success("Email updated. Sign in again with your new address."), vm.awaitLink("c-1"))
        assertNull(session.getAccessToken())
        assertNull(session.getRefreshToken())
        assertTrue(bodies("/api/v1/auth/refresh").single().contains(""""refreshToken":"refresh-1""""))
        assertTrue(bodies("/api/v1/auth/confirm-email-change").single().contains(""""token":"c-1""""))

        respond = { _, _ ->
            MockResponse().setResponseCode(409).setBody(
                """{"success":false,"errors":[{"code":"EMAIL_TAKEN","message":"That email was registered to another account before you confirmed."}]}""",
            )
        }
        session.saveSession("access-2", "RUNNER", "Juan", "juan@x.com", refreshToken = "refresh-2")
        vm.confirmEmailChange("c-2")
        assertEquals(EmailLinkState.Error("That email was registered to another account before you confirmed."), vm.awaitLink("c-2"))
        assertEquals("access-2", session.getAccessToken())
    }

    @Test fun `a confirmed email change still reads as confirmed, credentials gone, when the rest of the local sign-out throws`() {
        respond = { path, _ ->
            if (path == "/api/v1/auth/refresh") {
                MockResponse().setResponseCode(401).setBody(
                    """{"success":false,"errors":[{"code":"INVALID_REFRESH_TOKEN","message":"Refresh token expired or revoked"}]}""",
                )
            } else {
                ok(CONFIRMED)
            }
        }
        // A phone whose tether prefs cannot be opened: clearSession() throws partway.
        val broken = object : ContextWrapper(application) {
            override fun getApplicationContext(): Context = this
            override fun getSharedPreferences(name: String, mode: Int): SharedPreferences =
                if (name == "quickpitik_tether") throw IllegalStateException("disk full") else super.getSharedPreferences(name, mode)
        }
        val failing = SessionManager::class.java.getDeclaredConstructor(Context::class.java)
            .apply { isAccessible = true }.newInstance(broken)
        failing.saveSession("access-1", "RUNNER", "Juan", "juan@x.com", refreshToken = "refresh-1")
        // RetrofitClient builds its one HTTP client around whichever SessionManager exists
        // at first use; build it against the real one now, so the swap cannot outlive this test.
        RetrofitClient.apiService
        val singleton = SessionManager::class.java.getDeclaredField("INSTANCE").apply { isAccessible = true }
        singleton.set(null, failing)
        try {
            val vm = AuthViewModel(application)
            vm.confirmEmailChange("c-1")
            assertEquals(EmailLinkState.Success("Email updated. Sign in again with your new address."), vm.awaitLink("c-1"))
            assertNull(failing.getAccessToken())
            // The prefs store, where a phone with a keystore keeps the tokens, was wiped before the throw.
            assertNull(failing.getUserEmail())
        } finally {
            singleton.set(null, session)
        }
    }

    // MS-7 finding L2: the change is confirmed on the server, so a session store
    // that throws while this phone checks whose account it was must not turn it
    // into "Confirmation Failed".
    @Test fun `a confirmed email change still reads as confirmed when reading the session throws`() {
        respond = { path, _ -> if (path == "/api/v1/auth/refresh") ok(REFRESHED) else ok(CONFIRMED) }
        var unreadable = false
        val broken = object : ContextWrapper(application) {
            override fun getApplicationContext(): Context = this
            override fun getSharedPreferences(name: String, mode: Int): SharedPreferences {
                val real = super.getSharedPreferences(name, mode)
                return object : SharedPreferences by real {
                    override fun getString(key: String?, defValue: String?): String? =
                        if (unreadable && key == "user_email") throw IllegalStateException("keystore locked") else real.getString(key, defValue)
                }
            }
        }
        val failing = SessionManager::class.java.getDeclaredConstructor(Context::class.java)
            .apply { isAccessible = true }.newInstance(broken)
        failing.saveSession("access-1", "RUNNER", "Juan", "juan@x.com", refreshToken = "refresh-1")
        // Same singleton swap as the test above, for the same reason.
        RetrofitClient.apiService
        val singleton = SessionManager::class.java.getDeclaredField("INSTANCE").apply { isAccessible = true }
        singleton.set(null, failing)
        try {
            val vm = AuthViewModel(application)
            unreadable = true
            vm.confirmEmailChange("c-1")
            assertEquals(EmailLinkState.Success("Email updated. Sign in again with your new address."), vm.awaitLink("c-1"))
        } finally {
            singleton.set(null, session)
        }
    }

    // Security audit 2026-10-14, MOB-1: the confirm route is public, so the link
    // can belong to any account. Only the changed account's refresh is refused.
    @Test fun `a confirmed email change for another account leaves this phone signed in`() {
        respond = { path, _ -> if (path == "/api/v1/auth/refresh") ok(REFRESHED) else ok(CONFIRMED) }
        session.saveSession("access-1", "RUNNER", "Juan", "juan@x.com", refreshToken = "refresh-1")
        val vm = AuthViewModel(application)

        vm.confirmEmailChange("c-1")

        assertTrue(vm.awaitLink("c-1") is EmailLinkState.Success)
        assertEquals("access-9", session.getAccessToken())
        assertEquals("refresh-9", session.getRefreshToken())
        assertTrue(bodies("/api/v1/auth/refresh").single().contains(""""refreshToken":"refresh-1""""))
    }

    @Test fun `no answer from the refresh keeps this phone signed in`() {
        respond = { path, _ -> if (path == "/api/v1/auth/refresh") MockResponse().setResponseCode(503) else ok(CONFIRMED) }
        session.saveSession("access-1", "RUNNER", "Juan", "juan@x.com", refreshToken = "refresh-1")
        val vm = AuthViewModel(application)

        vm.confirmEmailChange("c-1")

        assertTrue(vm.awaitLink("c-1") is EmailLinkState.Success)
        assertEquals("access-1", session.getAccessToken())
    }

    @Test fun `a confirm while signed out sends no refresh`() {
        respond = { _, _ -> ok(CONFIRMED) }
        val vm = AuthViewModel(application)

        vm.confirmEmailChange("c-1")

        assertTrue(vm.awaitLink("c-1") is EmailLinkState.Success)
        assertTrue(bodies("/api/v1/auth/refresh").isEmpty())
    }

    @Test fun `an answered link outlives process death, and one that died in flight is sent again`() {
        val release = CountDownLatch(1)
        respond = { _, body ->
            if (body.contains("v-2")) release.await(5, TimeUnit.SECONDS)
            ok("""{"message":"Email confirmed. Thanks!"}""")
        }
        val saved = SavedStateHandle()
        val vm = AuthViewModel(application, saved)
        vm.verifyEmail("v-1")
        val answer = vm.awaitLink("v-1")
        vm.verifyEmail("v-2") // the process dies before this one is answered
        requireNotNull(backend.server.takeRequest(5, TimeUnit.SECONDS)) // v-1
        requireNotNull(backend.server.takeRequest(5, TimeUnit.SECONDS)) // v-2, first life

        val reborn = AuthViewModel(application, afterProcessDeath(saved))
        reborn.verifyEmail("v-1")
        reborn.verifyEmail("v-2")
        assertEquals(answer, reborn.emailLinks.value["v-1"])
        assertNotNull("v-2 must be sent again", backend.server.takeRequest(5, TimeUnit.SECONDS))
        release.countDown()
        assertEquals(EmailLinkState.Success("Email confirmed. Thanks!"), reborn.awaitLink("v-2"))
        assertNull("v-1 must not be sent again", backend.server.takeRequest(500, TimeUnit.MILLISECONDS))
    }

    private companion object {
        const val AUTH = """{"accessToken":"access-1","refreshToken":"refresh-1","user":{"id":"u1","email":"juan@x.com","name":"Juan","role":"RUNNER","createdAt":"2026-09-11T00:00:00Z"}}"""
        const val BLANK_AUTH = """{"accessToken":"","refreshToken":"refresh-1","user":{"id":"u1","email":"juan@x.com","name":"Juan","role":"RUNNER","createdAt":"2026-09-11T00:00:00Z"}}"""
        const val AUTH_PHOTOGRAPHER = """{"accessToken":"access-p","refreshToken":"refresh-p","user":{"id":"u2","email":"juan@x.com","name":"Juan","role":"PHOTOGRAPHER","createdAt":"2026-09-11T00:00:00Z"}}"""
        const val CONFIRMED = """{"message":"Email updated. Sign in again with your new address."}"""
        const val REFRESHED = """{"accessToken":"access-9","refreshToken":"refresh-9","user":{"id":"u1","email":"juan@x.com","name":"Juan","role":"RUNNER","avatarUrl":null,"createdAt":"2026-01-01T00:00:00Z"}}"""
    }
}
