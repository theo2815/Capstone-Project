package com.quickpitik.mobile.data.local

import android.content.Context
import android.content.SharedPreferences

// THE role check — every "is this a photographer?" branch goes through here.
// Three call sites used to disagree (equals("PHOTOGRAPHER") in MainActivity vs
// contains("PHOTO") on the auth screens), which would route a hypothetical new
// role differently at login than at cold start. Mobile only ever branches
// photographer/not-photographer, so a boolean beats an enum.
fun isPhotographerRole(raw: String?): Boolean =
    raw.equals("PHOTOGRAPHER", ignoreCase = true)

class SessionManager private constructor(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val PREF_NAME = "quickpitik_session"
        private const val KEY_ACCESS_TOKEN = "access_token"
        // F7 (2026-05-27): persist the refresh token so a 401 from the
        // 15-min access-token TTL can recover via POST /auth/refresh
        // instead of force-logging the user out.
        private const val KEY_REFRESH_TOKEN = "refresh_token"
        private const val KEY_USER_ROLE = "user_role"
        private const val KEY_USER_NAME = "user_name"
        private const val KEY_USER_EMAIL = "user_email"
        private const val KEY_USER_AVATAR = "user_avatar"

        @Volatile
        private var INSTANCE: SessionManager? = null

        fun getInstance(context: Context): SessionManager {
            return INSTANCE ?: synchronized(this) {
                val instance = SessionManager(context.applicationContext)
                INSTANCE = instance
                instance
            }
        }
    }

    fun saveSession(
        token: String,
        role: String,
        name: String,
        email: String,
        avatarUrl: String? = null,
        refreshToken: String? = null,
    ) {
        prefs.edit().apply {
            putString(KEY_ACCESS_TOKEN, token)
            putString(KEY_REFRESH_TOKEN, refreshToken)
            putString(KEY_USER_ROLE, role)
            putString(KEY_USER_NAME, name)
            putString(KEY_USER_EMAIL, email)
            putString(KEY_USER_AVATAR, avatarUrl)
            apply()
        }
    }

    // Token-only writer for the 401-refresh path. saveSession() would blank the
    // cached role/name/email/avatar (its params are required), and the refresh
    // response isn't a reliable place to re-derive them — a locally edited name
    // would be silently reverted. TokenAuthenticator calls this instead.
    fun updateTokens(accessToken: String, refreshToken: String?) {
        prefs.edit().apply {
            putString(KEY_ACCESS_TOKEN, accessToken)
            if (refreshToken != null) putString(KEY_REFRESH_TOKEN, refreshToken)
            apply()
        }
    }

    fun saveUserName(name: String) {
        prefs.edit().putString(KEY_USER_NAME, name).apply()
    }

    fun saveAvatarUrl(url: String?) {
        prefs.edit().putString(KEY_USER_AVATAR, url).apply()
    }

    fun getAccessToken(): String? {
        return prefs.getString(KEY_ACCESS_TOKEN, null)
    }

    fun getRefreshToken(): String? {
        return prefs.getString(KEY_REFRESH_TOKEN, null)
    }

    fun getUserRole(): String? {
        return prefs.getString(KEY_USER_ROLE, null)
    }

    fun getUserName(): String? {
        return prefs.getString(KEY_USER_NAME, null)
    }

    fun getUserEmail(): String? {
        return prefs.getString(KEY_USER_EMAIL, null)
    }

    fun getAvatarUrl(): String? {
        return prefs.getString(KEY_USER_AVATAR, null)
    }

    fun clearSession() {
        prefs.edit().clear().apply()
    }
}
