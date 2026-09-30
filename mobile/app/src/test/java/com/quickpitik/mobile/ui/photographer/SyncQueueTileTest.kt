package com.quickpitik.mobile.ui.photographer

import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.UploadRecord
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

@RunWith(RobolectricTestRunner::class)
class SyncQueueTileTest {
    @get:Rule val compose = createComposeRule()

    @Test fun `missing queued or failed original never claims a cloud copy`() {
        val missing = File(ApplicationProvider.getApplicationContext<android.content.Context>().filesDir,
            "missing-m10-${System.nanoTime()}.jpg")
        val record = mutableStateOf(UploadRecord(filePath = missing.absolutePath, eventId = "event",
            photographerId = "owner", captureTimestamp = 1, uploadStatus = "QUEUED"))
        compose.setContent { SyncQueueTile(record.value) }
        compose.onNodeWithText("No file").assertIsDisplayed()
        compose.onNodeWithText("Cloud").assertDoesNotExist()
        compose.runOnIdle { record.value = record.value.copy(uploadStatus = "FAILED") }
        compose.onNodeWithText("No file").assertIsDisplayed()
        compose.onNodeWithText("Cloud").assertDoesNotExist()
    }
}
