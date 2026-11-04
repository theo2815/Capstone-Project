package com.quickpitik.mobile.data.local

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/** The face-matching tick is kept per device (privacy plan D2) and asked again when the notice changes. */
@RunWith(RobolectricTestRunner::class)
class FaceConsentPrefsTest {
    private val context: Context = ApplicationProvider.getApplicationContext()

    @Test fun `accepting stores the current notice version`() {
        val prefs = FaceConsentPrefs(context)
        assertNull(prefs.acceptedVersion)
        prefs.accept()
        assertEquals(FACE_NOTICE_VERSION, FaceConsentPrefs(context).acceptedVersion)
    }

    @Test fun `signing out keeps the tick`() {
        FaceConsentPrefs(context).accept()
        SessionManager.getInstance(context).clearSession()
        assertEquals(FACE_NOTICE_VERSION, FaceConsentPrefs(context).acceptedVersion)
    }

    @Test fun `notice is needed until the current version is ticked`() {
        assertTrue(faceNoticeNeeded(null))
        assertFalse(faceNoticeNeeded(FACE_NOTICE_VERSION))
        assertTrue(faceNoticeNeeded(accepted = 1, current = 2))
        assertTrue(faceNoticeNeeded(accepted = 1)) // the notice was reworded (PT-0b): version-1 ticks are asked again
    }
}
