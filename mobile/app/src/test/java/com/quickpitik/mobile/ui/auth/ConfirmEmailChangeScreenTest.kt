package com.quickpitik.mobile.ui.auth

import android.app.Application
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.lifecycle.SavedStateHandle
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import org.junit.After
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * The confirm screen after a confirmed email change (security audit 2026-10-14,
 * MOB-1): a phone that stayed signed in is not told to sign in again. The answer
 * is already on file, so the screen sends nothing.
 */
@RunWith(RobolectricTestRunner::class)
class ConfirmEmailChangeScreenTest {
    @get:Rule val compose = createComposeRule()

    private val application: Application = ApplicationProvider.getApplicationContext()
    private val session = SessionManager.getInstance(application)

    @After fun tearDown() = session.clearSession()

    private fun answered() = AuthViewModel(
        application,
        SavedStateHandle(mapOf("emailLinks" to mapOf("c-1" to EmailLinkState.Success(CONFIRMED)))),
    )

    private fun waitFor(text: String) =
        compose.waitUntil(5_000) { compose.onAllNodesWithText(text).fetchSemanticsNodes().isNotEmpty() }

    @Test fun `a confirmed change that left this phone signed in says so and goes back`() {
        session.saveSession("access-1", "RUNNER", "Juan", "juan@x.com", refreshToken = "refresh-1")
        compose.setContent { ConfirmEmailChangeScreen(answered(), "c-1") {} }

        waitFor("Go back")
        compose.onNodeWithText("Email change confirmed. This phone is still signed in.").assertExists()
        compose.onAllNodesWithText("Sign In Again").assertCountEquals(0)
    }

    @Test fun `a confirmed change that signed this phone out asks to sign in again`() {
        session.clearSession()
        compose.setContent { ConfirmEmailChangeScreen(answered(), "c-1") {} }

        waitFor("Sign In Again")
        compose.onNodeWithText(CONFIRMED).assertExists()
    }

    private companion object {
        const val CONFIRMED = "Email updated. Sign in again with your new address."
    }
}
