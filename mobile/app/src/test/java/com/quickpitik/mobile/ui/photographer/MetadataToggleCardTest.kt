package com.quickpitik.mobile.ui.photographer

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.performClick
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class MetadataToggleCardTest {
    @get:Rule val compose = createComposeRule()

    @Test fun `metadata choice is one labeled checkbox action and disables during save`() {
        val label = "Show metadata publicly for new events"
        var changes = 0
        val saving = mutableStateOf(false)
        compose.setContent {
            var checked by remember { mutableStateOf(false) }
            MetadataToggleCard(
                state = MetadataToggleState(saved = false, draft = checked, loading = false, saving = saving.value),
                label = label,
                help = "New events only.",
                saveLabel = "Save metadata default",
                onChange = { checked = it; changes++ },
                onSave = {},
                onCancel = {},
                onRetry = {},
            )
        }
        val checkbox = SemanticsMatcher.expectValue(SemanticsProperties.Role, Role.Checkbox) and hasText(label)
        compose.onAllNodes(hasClickAction(), useUnmergedTree = true).assertCountEquals(1)
        compose.onNode(checkbox).assertIsOff().performClick()
        compose.onNode(checkbox).assertIsOn()
        compose.runOnIdle { assertEquals(1, changes); saving.value = true }
        compose.onNode(checkbox).assertIsNotEnabled()
    }
}
