package com.quickpitik.mobile.data.local

import android.content.Context

/**
 * The three `quickpitik_tether` keys, in one place.
 *
 * They record that a shutter watch is live and for whom, so a relaunch after
 * process death resumes it instead of waiting for a tap nobody knows is
 * needed, and they hold the event as JSON so that resume works offline.
 *
 * Three writers share them - [com.quickpitik.mobile.data.usb.CameraIngestCoordinator]
 * on start and stop, the photographer dashboard ViewModel for the snapshot and
 * the resume reads, and [SessionManager.clearSession] on sign-out - and each
 * used to re-type the literals.
 */
class TetherWatchPrefs(context: Context) {
    private val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    val activeEventId: String? get() = prefs.getString(KEY_EVENT, null)
    val owner: String? get() = prefs.getString(KEY_OWNER, null)
    val eventSnapshot: String? get() = prefs.getString(KEY_SNAPSHOT, null)
    val hasActiveEvent: Boolean get() = prefs.contains(KEY_EVENT)

    fun save(owner: String, eventId: String) =
        prefs.edit().putString(KEY_EVENT, eventId).putString(KEY_OWNER, owner).apply()

    fun saveSnapshot(json: String) = prefs.edit().putString(KEY_SNAPSHOT, json).apply()

    /** Removes the three keys, not the whole file - as every caller did by hand. */
    fun clear() = prefs.edit().remove(KEY_EVENT).remove(KEY_OWNER).remove(KEY_SNAPSHOT).apply()

    private companion object {
        const val PREF_NAME = "quickpitik_tether"
        const val KEY_EVENT = "watch_active_event"
        const val KEY_OWNER = "watch_owner"
        const val KEY_SNAPSHOT = "watch_event_snapshot"
    }
}
