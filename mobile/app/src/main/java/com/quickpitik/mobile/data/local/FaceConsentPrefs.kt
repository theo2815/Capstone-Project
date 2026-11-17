package com.quickpitik.mobile.data.local

import android.content.Context

/**
 * Raise to ask everyone again after the face-matching notice changes (website: FACE_NOTICE_VERSION).
 * 2 since 2026-11-05: the notice names Singapore and covers a child's selfie (review S10).
 */
const val FACE_NOTICE_VERSION = 2

/** The bib-search notice (task BT-2, review S10; website: BIB_NOTICE_VERSION); raising it asks everyone again. */
const val BIB_NOTICE_VERSION = 1

internal fun faceNoticeNeeded(accepted: Int?, current: Int = FACE_NOTICE_VERSION): Boolean = accepted != current

/**
 * The face-matching tick (privacy plan D2) and the bib-search tick (BT-2),
 * under separate keys: kept on the device, so guests are covered too; no
 * server record. Not cleared at sign-out, so [SessionManager.clearSession]
 * leaves them alone.
 */
class FaceConsentPrefs(context: Context) {
    private val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    val acceptedVersion: Int? get() = version(KEY_VERSION)
    val bibAcceptedVersion: Int? get() = version(KEY_BIB_VERSION)

    fun accept() = prefs.edit().putInt(KEY_VERSION, FACE_NOTICE_VERSION).apply()
    fun acceptBib() = prefs.edit().putInt(KEY_BIB_VERSION, BIB_NOTICE_VERSION).apply()

    private fun version(key: String): Int? = if (prefs.contains(key)) prefs.getInt(key, 0) else null

    private companion object {
        const val PREF_NAME = "quickpitik_face_consent"
        const val KEY_VERSION = "accepted_version"
        const val KEY_BIB_VERSION = "bib_accepted_version"
    }
}
