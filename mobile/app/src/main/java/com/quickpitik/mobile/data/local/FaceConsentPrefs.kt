package com.quickpitik.mobile.data.local

import android.content.Context

/** Raise to ask everyone again after the face-matching notice changes (website: FACE_NOTICE_VERSION). */
const val FACE_NOTICE_VERSION = 1

internal fun faceNoticeNeeded(accepted: Int?, current: Int = FACE_NOTICE_VERSION): Boolean = accepted != current

/**
 * The face-matching tick (privacy plan D2): kept on the device, so guests are
 * covered too; no server record. Not cleared at sign-out, so
 * [SessionManager.clearSession] leaves it alone.
 */
class FaceConsentPrefs(context: Context) {
    private val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    val acceptedVersion: Int? get() = if (prefs.contains(KEY_VERSION)) prefs.getInt(KEY_VERSION, 0) else null

    fun accept() = prefs.edit().putInt(KEY_VERSION, FACE_NOTICE_VERSION).apply()

    private companion object {
        const val PREF_NAME = "quickpitik_face_consent"
        const val KEY_VERSION = "accepted_version"
    }
}
