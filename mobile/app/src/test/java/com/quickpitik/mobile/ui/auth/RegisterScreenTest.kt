package com.quickpitik.mobile.ui.auth

import android.app.Application
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTextInput
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.ui.common.SIGNUP_TICK_ERROR
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/** PT-4b: a valid form without the agreement tick shows the error and sends nothing. */
@RunWith(RobolectricTestRunner::class)
class RegisterScreenTest {
    @get:Rule val compose = createComposeRule()
    @get:Rule val backend = MockBackend()

    @Test fun `create account without the agreement tick shows the error and sends no request`() {
        val application: Application = ApplicationProvider.getApplicationContext()
        compose.setContent { RegisterScreen(AuthViewModel(application), {}, {}) }

        compose.onNodeWithText("Juan dela Cruz").performTextInput("Juan")
        compose.onNodeWithText("you@example.com").performTextInput("juan@x.com")
        // Each filled field drops its placeholder, so the next one is index 0 again.
        compose.onAllNodesWithText("••••••••")[0].performTextInput("secret123")
        compose.onAllNodesWithText("••••••••")[0].performTextInput("secret123")
        compose.onAllNodesWithText("CREATE ACCOUNT")[1].performScrollTo().performClick()

        compose.waitUntil(5_000) { compose.onAllNodesWithText(SIGNUP_TICK_ERROR).fetchSemanticsNodes().isNotEmpty() }
        assertEquals(0, backend.server.requestCount)
    }
}
