package com.quickpitik.mobile.ui.common

import android.app.Application
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.runComposeUiTest
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/** Website report-dialog parity (AA-R17a): signed-in only, note 10-500, a failure keeps the note. */
@RunWith(RobolectricTestRunner::class)
@OptIn(ExperimentalTestApi::class)
class ReportActionTest {
    @get:Rule val backend = MockBackend()
    private val application: Application = ApplicationProvider.getApplicationContext()

    @After fun tearDown() = SessionManager.getInstance(application).clearSession()

    @Test fun `note length rule`() {
        assertFalse(canSendReport("too short"))
        assertFalse(canSendReport("   short    "))
        assertTrue(canSendReport("ten chars!"))
        assertTrue(canSendReport("x".repeat(500)))
        assertFalse(canSendReport("x".repeat(501)))
    }

    @Test fun `hidden when signed out`() = runComposeUiTest {
        SessionManager.getInstance(application).clearSession()
        setContent { ReportAction(ReportTarget.Event("e1")) }
        onNodeWithText("Report").assertDoesNotExist()
    }

    @Test fun `a short note cannot be sent and a failure keeps the note`() {
        signInAs("RUNNER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = MockResponse().setResponseCode(429).setBody(
                """{"success":false,"errors":[{"code":"RATE_LIMITED","message":"Too many reports. Try again later."}]}""",
            )
        }
        runComposeUiTest {
            setContent { ReportAction(ReportTarget.Photographer("juan")) }
            onNodeWithText("Report").performClick()
            onNodeWithText("Send report").assertIsNotEnabled()
            onNodeWithText("Tell us what is wrong", substring = true).performTextInput("Fake photos, not his.")
            onNodeWithText("Send report").performClick()
            waitUntil(timeoutMillis = 5_000) { onAllNodesWithText("Too many reports. Try again later.").fetchSemanticsNodes().isNotEmpty() }
            onNodeWithText("Fake photos, not his.").assertExists()
        }
        assertEquals("/api/v1/me/reports/photographers/juan", backend.server.takeRequest().path)
    }

    @Test fun `thanks the user also when already reported`() {
        signInAs("RUNNER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest) = ok("""{"reported":true,"alreadyReported":true}""")
        }
        runComposeUiTest {
            setContent { ReportAction(ReportTarget.Event("e1")) }
            onNodeWithText("Report").performClick()
            onNodeWithText("Tell us what is wrong", substring = true).performTextInput("  This is not a real race.  ")
            onNodeWithText("Send report").performClick()
            waitUntil(timeoutMillis = 5_000) { onAllNodesWithText("Thanks, an admin will look at it.").fetchSemanticsNodes().isNotEmpty() }
        }
        val sent = backend.server.takeRequest()
        assertEquals("/api/v1/me/reports/events/e1", sent.path)
        assertEquals("""{"note":"This is not a real race."}""", sent.body.readUtf8())
    }
}
