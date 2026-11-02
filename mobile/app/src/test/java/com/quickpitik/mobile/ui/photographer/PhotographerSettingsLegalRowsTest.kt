package com.quickpitik.mobile.ui.photographer

import android.app.Application
import android.content.Intent
import androidx.compose.ui.test.hasScrollAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollToNode
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import com.quickpitik.mobile.ui.common.PRIVACY_URL
import com.quickpitik.mobile.ui.common.SUPPORT_EMAIL
import com.quickpitik.mobile.ui.common.TERMS_URL
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows

/**
 * The photographer's settings end with the same legal rows as the runner's:
 * the website's Privacy and Terms pages, the support email, and a deletion
 * request that carries the signed-in photographer's email. The brand fetch
 * must succeed, or the screen shows its load error instead of the form.
 */
@RunWith(RobolectricTestRunner::class)
class PhotographerSettingsLegalRowsTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    @get:Rule val compose = createComposeRule()

    private lateinit var application: Application

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse =
                if (request.path.orEmpty().substringBefore('?') == "/api/v1/me/photographer/brand") {
                    ok(BRAND)
                } else {
                    MockResponse().setResponseCode(404)
                }
        }
        val dashboard = PhotographerDashboardViewModel(application)
        val settings = StudioSettingsViewModel(application)
        compose.setContent { PhotographerSettingsScreen(dashboard, settings, onLogout = {}) }
    }

    private val BRAND = """{"brandName":"Juan Photos","brandColor":"amber","bio":"","handle":"juan","regionCode":"07","provinceCode":"0722","coverUrl":null,"watermarkUrl":null,"avatarUrl":null,"watermarkSettings":{"opacity":0.7,"scale":0.15,"landscapeX":1.0,"landscapeY":1.0,"portraitX":1.0,"portraitY":1.0}}"""

    @After fun tearDown() = SessionManager.getInstance(application).clearSession()

    private fun scrollTo(label: String) {
        compose.onNode(hasScrollAction()).performScrollToNode(hasText(label))
    }

    private fun tap(label: String): Intent {
        scrollTo(label)
        compose.onNodeWithText(label).performClick()
        return Shadows.shadowOf(application).nextStartedActivity
    }

    @Test fun `privacy and terms rows open the website pages`() {
        assertEquals(PRIVACY_URL, tap("Privacy Policy").dataString)
        assertEquals(TERMS_URL, tap("Terms of Service").dataString)
    }

    @Test fun `support email is shown`() {
        scrollTo(SUPPORT_EMAIL)
    }

    @Test fun `deletion row emails support with the photographer's account email`() {
        val intent = tap("Request account deletion")

        assertEquals(Intent.ACTION_SENDTO, intent.action)
        assertTrue(intent.dataString, intent.dataString!!.contains("juan%40x.com"))
    }
}
