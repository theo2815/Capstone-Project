package com.quickpitik.mobile.ui.runner

import android.content.Context
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performImeAction
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.BIB_NOTICE_VERSION
import com.quickpitik.mobile.data.local.FaceConsentPrefs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Review S10 (task BT-2): a bib search looks for a person's photos too, so no
 * bib search runs before the runner (or guest) ticks the bib notice once on
 * this device, whether they tap the button or the keyboard's search key.
 */
@RunWith(RobolectricTestRunner::class)
class BibPanelBibNoticeTest {
    @get:Rule val compose = createComposeRule()
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val calls = mutableListOf<String>()
    private val tick = "I agree. This is my bib number, my child's, or I have the runner's permission."

    @Before fun setUp() {
        context.getSharedPreferences("quickpitik_face_consent", Context.MODE_PRIVATE).edit().clear().commit()
        compose.setContent {
            SearchPanel(
                mode = SearchPanelMode.Bib,
                bib = "1234",
                photoCount = 0,
                selfie = SelfiePanelState(emptyList(), saveToLibrary = true, canSave = false, saveNotice = null, matching = false, error = null),
                callbacks = SearchPanelCallbacks(
                    onBibChange = {}, onSubmitBib = { calls += "bib" }, onSwitchToSelfie = {}, onSwitchToBib = {},
                    onTakeSelfie = {}, onUploadSelfie = {},
                    onMatchAllSelfies = {}, onPickSelfie = {}, onSaveToLibraryChange = {},
                ),
            )
        }
    }

    @Test fun `bib search stays off until the bib box is ticked`() {
        compose.onNodeWithText("we look for this bib number in this event's photos", substring = true).assertExists()
        compose.onNodeWithText("How search works", substring = true).assertExists()
        compose.waitForIdle()
        compose.onNode(hasClickAction() and hasText("Search by bib", substring = true)).assertIsNotEnabled().performClick()
        compose.onNode(hasSetTextAction()).performImeAction()
        assertEquals(emptyList<String>(), calls)

        compose.onNodeWithText(tick).performClick()
        compose.waitForIdle()
        compose.onNodeWithText(tick).assertDoesNotExist()
        compose.onNode(hasClickAction() and hasText("Search by bib", substring = true)).assertIsEnabled().performClick()
        compose.onNode(hasSetTextAction()).performImeAction()
        assertEquals(listOf("bib", "bib"), calls)
        assertEquals(BIB_NOTICE_VERSION, FaceConsentPrefs(context).bibAcceptedVersion)
        assertNull(FaceConsentPrefs(context).acceptedVersion) // the bib tick is not a face-matching tick
    }
}
