package com.quickpitik.mobile.ui.runner

import android.content.Context
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.FACE_NOTICE_VERSION
import com.quickpitik.mobile.data.local.FaceConsentPrefs
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Privacy plan D2: no selfie reaches face matching before the runner (or
 * guest) ticks the face-matching notice once on this device.
 */
@RunWith(RobolectricTestRunner::class)
class SelfiePanelFaceNoticeTest {
    @get:Rule val compose = createComposeRule()
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val calls = mutableListOf<String>()

    @Before fun setUp() {
        context.getSharedPreferences("quickpitik_face_consent", Context.MODE_PRIVATE).edit().clear().commit()
        compose.setContent {
            SearchPanel(
                mode = SearchPanelMode.Selfie,
                bib = "",
                photoCount = 0,
                selfie = SelfiePanelState(emptyList(), saveToLibrary = true, canSave = false, saveNotice = null, matching = false, error = null),
                callbacks = SearchPanelCallbacks(
                    onBibChange = {}, onSubmitBib = {}, onSwitchToSelfie = {}, onSwitchToBib = {},
                    onTakeSelfie = { calls += "take" }, onUploadSelfie = { calls += "upload" },
                    onMatchAllSelfies = {}, onPickSelfie = {}, onSaveToLibraryChange = {},
                ),
            )
        }
    }

    @Test fun `selfie actions stay off until the face-matching box is ticked`() {
        compose.onNodeWithText("Amazon Rekognition", substring = true).assertExists()
        compose.onNodeWithText("Singapore", substring = true).assertExists()
        compose.onNodeWithText("How face matching works", substring = true).assertExists()
        compose.waitForIdle()
        compose.onNode(hasClickAction() and hasText("Take a selfie", substring = true)).assertIsNotEnabled().performClick()
        compose.onNode(hasClickAction() and hasText("Upload a selfie", substring = true)).assertIsNotEnabled().performClick()
        assertEquals(emptyList<String>(), calls)

        compose.onNodeWithText("I agree to face matching with this selfie. If it shows my child, I am their parent or guardian.").performClick()
        compose.waitForIdle()
        compose.onNodeWithText("I agree to face matching with this selfie. If it shows my child, I am their parent or guardian.").assertDoesNotExist()
        compose.onNode(hasClickAction() and hasText("Take a selfie", substring = true)).assertIsEnabled().performClick()
        compose.onNode(hasClickAction() and hasText("Upload a selfie", substring = true)).performClick()
        assertEquals(listOf("take", "upload"), calls)
        assertEquals(FACE_NOTICE_VERSION, FaceConsentPrefs(context).acceptedVersion)
    }
}
