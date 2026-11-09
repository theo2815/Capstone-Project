package com.quickpitik.mobile.ui.runner

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
import com.quickpitik.mobile.ui.common.PHOTO_DELETION_URL
import com.quickpitik.mobile.ui.common.PRIVACY_URL
import com.quickpitik.mobile.ui.common.SUPPORT_EMAIL
import com.quickpitik.mobile.ui.common.TERMS_URL
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.Shadows

/**
 * The runner's settings rows open the website's legal pages, and the deletion
 * row opens the mail app with the signed-in runner's email already in it, so
 * support knows which account to delete.
 */
// A phone-sized screen: the default 320x470 cannot lay out the whole settings card once it has this many rows.
@Config(qualifiers = "w411dp-h891dp")
@RunWith(RobolectricTestRunner::class)
class AccountSettingsLegalRowsTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    @get:Rule val compose = createComposeRule()

    private lateinit var application: Application

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        SessionManager.getInstance(application).apply {
            clearSession()
            saveSession("t", "RUNNER", "Juan", "juan@x.com", refreshToken = "rt-1")
        }
        compose.setContent { AccountSettingsScreen(ProfileViewModel(application), onLogout = {}) }
    }

    @After fun tearDown() = SessionManager.getInstance(application).clearSession()

    private fun tap(label: String): Intent {
        compose.onNode(hasScrollAction()).performScrollToNode(hasText(label))
        compose.onNodeWithText(label).performClick()
        return Shadows.shadowOf(application).nextStartedActivity
    }

    @Test fun `privacy and terms rows open the website pages`() {
        assertEquals(PRIVACY_URL, tap("Privacy Policy").dataString)
        assertEquals(TERMS_URL, tap("Terms of Service").dataString)
    }

    @Test fun `the photo deletion row opens the website request page`() {
        assertEquals(PHOTO_DELETION_URL, tap("Request photo deletion").dataString)
    }

    @Test fun `support email is shown`() {
        compose.onNode(hasScrollAction()).performScrollToNode(hasText(SUPPORT_EMAIL))
    }

    @Test fun `deletion row emails support with the runner's account email`() {
        val intent = tap("Request account deletion")

        assertEquals(Intent.ACTION_SENDTO, intent.action)
        assertTrue(intent.dataString, intent.dataString!!.contains("juan%40x.com"))
    }
}
