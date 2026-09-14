package com.quickpitik.mobile.ui.photographer

import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class SharePhotoTileTest {
    @get:Rule val compose = createComposeRule()

    @Test
    fun `processing badge follows the refreshed photo status`() {
        val status = mutableStateOf("processing")
        compose.setContent {
            SharePhotoTile(
                photo = PhotographerLibraryPhotoDto(
                    id = "photo-1", bib = null, status = status.value, salesCount = 0,
                    uploadedAt = "2026-09-14T00:00:00Z", tone = 0, span = "default",
                ),
                selecting = false, selected = false, onClick = {},
            )
        }
        compose.onNodeWithText("PROCESSING").assertIsDisplayed()
        compose.runOnIdle { status.value = "live" }
        compose.onNodeWithText("PROCESSING").assertDoesNotExist()
        compose.runOnIdle { status.value = "hidden" }
        compose.onNodeWithText("PROCESSING").assertDoesNotExist()
    }
}
