package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.UserDto
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.MockResponse
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/** RV-L6: the blocking "accept the updated Terms" prompt on the studio shell. */
@RunWith(RobolectricTestRunner::class)
class TermsAcceptanceTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application
    private val viewModels = ViewModelStore()

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
    }

    @After fun tearDown() {
        viewModels.clear()
        SessionManager.getInstance(application).clearSession()
    }

    private fun user(required: Boolean, version: String? = "2026-11-15") = """
        {"id":"u1","email":"juan@x.com","name":"Juan","role":"PHOTOGRAPHER","createdAt":"2026-01-01T00:00:00Z",
         "termsAcceptanceRequired":$required,"legalCurrentVersion":${version?.let { "\"$it\"" }}}
    """.trimIndent()

    private fun model() = TermsAcceptanceViewModel(application).also { viewModels.put("terms", it) }

    private fun dto(required: Boolean, version: String?) =
        UserDto(id = "u1", email = "a@b.c", name = "A", role = "PHOTOGRAPHER", createdAt = "x",
            termsAcceptanceRequired = required, legalCurrentVersion = version)

    @Test fun `the prompt is due only when the backend asks and names a version`() {
        assertTrue(mustAcceptTerms(dto(true, "v2")))
        assertFalse(mustAcceptTerms(dto(false, "v2")))
        assertFalse(mustAcceptTerms(dto(true, null)))
    }

    @Test fun `a user who must accept sees the prompt`() {
        backend.enqueue(ok(user(required = true)))
        val vm = model().also { it.load() }
        assertEquals("2026-11-15", vm.state.awaitFirst { it.required }.version)
    }

    @Test fun `a failed first load is retried and then shows the prompt`() {
        backend.enqueue(MockResponse().setResponseCode(500), ok(user(required = true)))
        val vm = model().also { it.load() }
        backend.server.takeRequest() // the failing load has reached the server
        vm.load()
        assertEquals("2026-11-15", vm.state.awaitFirst { it.required }.version)
    }

    @Test fun `accepting posts the version the backend named and clears the prompt`() {
        backend.enqueue(ok(user(required = true)), ok(user(required = false)))
        val vm = model().also { it.load() }
        vm.state.awaitFirst { it.required }
        vm.accept()
        vm.state.awaitFirst { !it.required }
        backend.server.takeRequest()
        val post = backend.server.takeRequest()
        assertEquals("POST", post.method)
        assertEquals("/api/v1/me/terms-acceptance", post.path)
        assertEquals("""{"version":"2026-11-15"}""", post.body.readUtf8())
    }

    @Test fun `a refused acceptance keeps the prompt, shows the backend's words and posts the new version next`() {
        backend.enqueue(
            ok(user(required = true, version = "v1")),
            MockResponse().setResponseCode(409).setBody(
                """{"success":false,"errors":[{"code":"CONSENT_VERSION_MISMATCH","message":"The Terms were updated."}]}""",
            ),
            ok(user(required = true, version = "v2")),
            ok(user(required = false, version = "v2")),
        )
        val vm = model().also { it.load() }
        vm.state.awaitFirst { it.required }
        vm.accept()
        val refused = vm.state.awaitFirst { it.version == "v2" }
        assertTrue(refused.required)
        assertEquals("The Terms were updated.", refused.error)
        vm.accept()
        vm.state.awaitFirst { !it.required }
        repeat(3) { backend.server.takeRequest() } // me, first POST, me again
        assertEquals("""{"version":"v2"}""", backend.server.takeRequest().body.readUtf8())
    }
}
