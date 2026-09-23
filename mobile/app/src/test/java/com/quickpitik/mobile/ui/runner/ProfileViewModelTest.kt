package com.quickpitik.mobile.ui.runner

import android.app.Application
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.ui.auth.validateEmail
import com.quickpitik.mobile.ui.auth.validateNewPassword
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
import org.robolectric.Shadows
import java.io.ByteArrayInputStream
import java.util.concurrent.CopyOnWriteArrayList

/**
 * The runner's account screen: the password change is honest about whether
 * this device stays signed in, the email change is only a request (the
 * address on file does not move), and name/avatar edits reach SessionManager,
 * which every other surface reads. Oversized picks never leave the phone.
 */
@RunWith(RobolectricTestRunner::class)
class ProfileViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private lateinit var session: SessionManager
    private val seen = CopyOnWriteArrayList<Pair<String, String>>() // "METHOD path" to body
    @Volatile private var password: MockResponse = ok("""{"message":"Password changed."}""")

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        session = SessionManager.getInstance(application)
        session.clearSession()
        session.saveSession("t", "RUNNER", "Juan", "juan@x.com", refreshToken = "rt-1")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val key = "${request.method} ${request.path}"
                seen += key to request.body.readUtf8()
                return when (key) {
                    "PUT /api/v1/me/password" -> password
                    "PUT /api/v1/me/email" -> ok("""{"message":"We sent a confirmation link to ana@x.com."}""")
                    "PUT /api/v1/me/profile" -> ok(user(name = "Juan Dela Cruz"))
                    "POST /api/v1/me/avatar" -> ok(user(avatar = "\"/avatars/a.jpg\""))
                    "DELETE /api/v1/me/avatar" -> ok(user())
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
    }

    @After fun tearDown() = session.clearSession()

    private fun user(name: String = "Juan", avatar: String = "null") =
        """{"id":"u1","email":"juan@x.com","name":"$name","role":"RUNNER","avatarUrl":$avatar,"createdAt":"2026-09-11T00:00:00Z"}"""

    private fun bodies(key: String) = seen.filter { it.first == key }.map { it.second }

    private fun picked(name: String, size: Int): Uri = Uri.parse("content://test/$name").also {
        Shadows.shadowOf(application.contentResolver).registerInputStream(it, ByteArrayInputStream(ByteArray(size)))
    }

    @Test fun `a password change spares this device only when there is a refresh token to spare`() {
        val vm = ProfileViewModel(application)
        vm.changePassword("old-pass", "")
        assertEquals("Passwords cannot be empty", vm.passwordUpdateError.value)
        vm.changePassword("old-pass", "short")
        assertEquals(validateNewPassword("short"), vm.passwordUpdateError.value)
        assertTrue(seen.isEmpty())

        vm.changePassword("old-pass", "N3w-Passw0rd!")
        vm.passwordUpdateSuccess.awaitFirst { it }
        assertTrue(vm.passwordSessionKept.value)
        assertTrue(bodies("PUT /api/v1/me/password").single().contains(""""refreshToken":"rt-1""""))

        // A session from before refresh tokens were stored: the backend signs out every device, this one too.
        session.saveSession("t2", "RUNNER", "Juan", "juan@x.com")
        vm.resetPasswordState()
        vm.changePassword("old-pass", "N3w-Passw0rd!")
        vm.passwordUpdateSuccess.awaitFirst { it }
        assertFalse(vm.passwordSessionKept.value)
        assertFalse(bodies("PUT /api/v1/me/password").last().contains("refreshToken"))

        password = MockResponse().setResponseCode(400)
            .setBody("""{"success":false,"errors":[{"code":"INVALID_CREDENTIALS","message":"Current password is wrong."}]}""")
        vm.resetPasswordState()
        vm.changePassword("wrong", "N3w-Passw0rd!")
        assertEquals("Current password is wrong.", vm.passwordUpdateError.awaitFirst { it != null })
        assertFalse(vm.passwordUpdateSuccess.value)
    }

    @Test fun `an email change is refused locally until it is real, and only ever requests the switch`() {
        val vm = ProfileViewModel(application)
        vm.requestEmailChange("not-an-email", "pw")
        assertEquals(validateEmail("not-an-email"), vm.emailChangeError.value)
        vm.requestEmailChange("ana@x.com", "")
        assertEquals("Enter your current password to confirm.", vm.emailChangeError.value)
        vm.requestEmailChange(" JUAN@X.COM ", "pw")
        assertEquals("That's already your sign-in email.", vm.emailChangeError.value)
        assertTrue(seen.isEmpty())

        vm.requestEmailChange(" ana@x.com ", "pw")
        assertEquals("We sent a confirmation link to ana@x.com.", vm.emailChangeMessage.awaitFirst { it != null })
        assertNull(vm.emailChangeError.value)
        assertEquals("juan@x.com", vm.profileEmail.value) // swaps only when the link is redeemed
        assertEquals("juan@x.com", session.getUserEmail())
        assertEquals(
            """{"newEmail":"ana@x.com","currentPassword":"pw"}""",
            bodies("PUT /api/v1/me/email").single(),
        )
    }

    @Test fun `name and avatar edits land in the session every other screen reads`() {
        val vm = ProfileViewModel(application)
        vm.updateName("   ")
        assertEquals("Name cannot be empty", vm.nameUpdateError.value)
        assertTrue(seen.isEmpty())

        vm.updateName("  Juan Dela Cruz ")
        vm.nameUpdateSuccess.awaitFirst { it }
        assertEquals("Juan Dela Cruz", vm.profileName.value)
        assertEquals("Juan Dela Cruz", session.getUserName())
        assertEquals("""{"name":"Juan Dela Cruz"}""", bodies("PUT /api/v1/me/profile").single())

        vm.uploadAvatar(picked("huge.jpg", MAX_UPLOAD_BYTES + 1))
        assertEquals("Profile photos must be 8 MB or smaller", vm.avatarError.awaitFirst { it != null })
        vm.uploadSelfie(picked("huge-selfie.jpg", MAX_UPLOAD_BYTES + 1))
        assertEquals("Selfies must be 8 MB or smaller", vm.selfiesError.awaitFirst { it != null })
        assertEquals(1, seen.size) // neither oversized pick was uploaded

        vm.uploadAvatar(picked("a.jpg", 16))
        assertEquals("/avatars/a.jpg", vm.avatarUrl.awaitFirst { it != null })
        assertEquals("/avatars/a.jpg", session.getAvatarUrl())

        vm.removeAvatar()
        vm.avatarUrl.awaitFirst { it == null }
        assertNull(session.getAvatarUrl())
    }
}
