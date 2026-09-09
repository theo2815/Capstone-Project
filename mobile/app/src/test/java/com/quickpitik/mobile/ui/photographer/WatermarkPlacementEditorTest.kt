package com.quickpitik.mobile.ui.photographer

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performSemanticsAction
import com.quickpitik.mobile.data.remote.WatermarkSettingsDto
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class WatermarkPlacementEditorTest {
    @get:Rule val compose = createComposeRule()

    @Test
    fun `accessibility actions position both orientations without dragging`() {
        var latest = WatermarkSettingsDto()
        compose.setContent {
            var placement by remember { mutableStateOf(latest) }
            WatermarkPlacementEditor(model = "", placement = placement) {
                placement = it
                latest = it
            }
        }
        compose.onNodeWithContentDescription("Horizontal position")
            .performSemanticsAction(SemanticsActions.SetProgress) { it(0.25f) }
        compose.onNodeWithContentDescription("Vertical position")
            .performSemanticsAction(SemanticsActions.SetProgress) { it(0.75f) }
        compose.runOnIdle { assertEquals(0.25 to 0.75, latest.pos(landscape = true)) }

        compose.onNodeWithText("Portrait").performClick()
        compose.onNodeWithContentDescription("Horizontal position")
            .performSemanticsAction(SemanticsActions.SetProgress) { it(0.5f) }
        compose.onNodeWithContentDescription("Vertical position")
            .performSemanticsAction(SemanticsActions.SetProgress) { it(0.0f) }
        compose.runOnIdle {
            assertEquals(0.5 to 0.0, latest.pos(landscape = false))
            assertEquals(0.25 to 0.75, latest.pos(landscape = true))
            assertEquals(0.15, latest.scale, 0.0)
            assertEquals(0.7, latest.opacity, 0.0)
        }
    }
}
