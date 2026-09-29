package com.quickpitik.mobile.data.local

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

// THE role check — every "is this a photographer?" branch goes through here.
// Three call sites used to disagree (equals("PHOTOGRAPHER") in MainActivity vs
// contains("PHOTO") on the auth screens), which would route a hypothetical new
// role differently at login than at cold start. Mobile only ever branches
// photographer/not-photographer, so a boolean beats an enum.
fun isPhotographerRole(raw: String?): Boolean =
    raw.equals("PHOTOGRAPHER", ignoreCase = true)

class SessionManager private constructor(context: Context) {
    private val appContext = context.applicationContext
    private var secureStorageAvailable = true
    @Volatile private var memoryAccessToken: String? = null
    @Volatile private var memoryRefreshToken: String? = null
    // The JWT + refresh token now live in an EncryptedSharedPreferences store
    // (AES-256) instead of the old plaintext prefs — a rooted/backed-up device
    // no longer leaks a bearer that hits the payout endpoints. Same
    // SharedPreferences interface, so every getter/setter below is unchanged.
    private val prefs: SharedPreferences = buildSecurePrefs(appContext)

    // Encrypted store with a one-time migration from the legacy plaintext file,
    // and memory-only tokens if the keystore fails. A restart then requires
    // sign-in; credentials must never fall back to plaintext persistence.
    private fun buildSecurePrefs(ctx: Context): SharedPreferences = try {
        val masterKey = MasterKey.Builder(ctx)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        val encrypted = EncryptedSharedPreferences.create(
            ctx,
            PREF_NAME_ENC,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
        migrateLegacyPrefs(ctx, encrypted)
        encrypted
    } catch (e: Exception) {
        android.util.Log.w(
            "SessionManager",
            "EncryptedSharedPreferences unavailable; session tokens will not persist",
            e,
        )
        secureStorageAvailable = false
        ctx.deleteSharedPreferences(PREF_NAME_ENC)
        ctx.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE).also {
            check(it.edit().clear().commit()) { "Could not clear legacy session storage" }
        }
    }

    // Copy tokens written by an older plaintext build into the encrypted store,
    // then wipe the plaintext file so the JWT stops sitting in cleartext. No-op
    // once migrated (legacy file empty).
    private fun migrateLegacyPrefs(ctx: Context, encrypted: SharedPreferences) {
        val legacy = ctx.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val all = legacy.all
        if (all.isEmpty()) return
        encrypted.edit().apply {
            for ((k, v) in all) {
                when (v) {
                    is String -> putString(k, v)
                    is Boolean -> putBoolean(k, v)
                    is Int -> putInt(k, v)
                    is Long -> putLong(k, v)
                    is Float -> putFloat(k, v)
                }
            }
            check(commit()) { "Could not migrate encrypted session" }
        }
        check(legacy.edit().clear().commit()) { "Could not clear legacy session storage" }
    }
    private var authGeneration = 0L
    private val _authChanges = MutableStateFlow(0L)
    val authChanges: StateFlow<Long> = _authChanges
    private val accessTokens = LinkedHashSet<String>()
    data class AuthSession(val owner: String, val generation: Long)

    @Synchronized
    fun authSnapshot(token: String): AuthSession? {
        if (token != getAccessToken() && token !in accessTokens) return null
        return getUserEmail()?.let { AuthSession(it, authGeneration) }
    }

    @Synchronized
    fun isCurrent(snapshot: AuthSession) = snapshot.generation == authGeneration && snapshot.owner == getUserEmail()

    @Synchronized
    fun refreshIfCurrent(snapshot: AuthSession, token: String, refresh: String?): Boolean {
        if (!isCurrent(snapshot)) return false
        updateTokens(token, refresh)
        return true
    }

    @Synchronized
    fun clearIfCurrent(snapshot: AuthSession): Boolean {
        if (!isCurrent(snapshot)) return false
        clearSession()
        return true
    }

    companion object {
        private const val PREF_NAME = "quickpitik_session"
        // Encrypted store (AES-256). Distinct file so the one-time migration from
        // the legacy plaintext PREF_NAME can detect + drain it.
        private const val PREF_NAME_ENC = "quickpitik_session_secure"
        private const val KEY_ACCESS_TOKEN = "access_token"
        // F7 (2026-05-27): persist the refresh token so a 401 from the
        // 15-min access-token TTL can recover via POST /auth/refresh
        // instead of force-logging the user out.
        private const val KEY_REFRESH_TOKEN = "refresh_token"
        private const val KEY_USER_ROLE = "user_role"
        private const val KEY_USER_NAME = "user_name"
        private const val KEY_USER_EMAIL = "user_email"
        private const val KEY_USER_AVATAR = "user_avatar"
        // Photographer→runner view switch (see ViewMode). Cleared by
        // clearSession()'s prefs.clear() and explicitly on saveSession so a
        // fresh login always lands in photographer view.
        private const val KEY_VIEW_MODE = "runner_view"

        @Volatile
        private var INSTANCE: SessionManager? = null

        fun getInstance(context: Context): SessionManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: SessionManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }

    @Synchronized
    fun saveSession(
        token: String,
        role: String,
        name: String,
        email: String,
        avatarUrl: String? = null,
        refreshToken: String? = null,
    ) {
        authGeneration++
        accessTokens.clear()
        memoryAccessToken = if (secureStorageAvailable) null else token
        memoryRefreshToken = if (secureStorageAvailable) null else refreshToken
        prefs.edit().apply {
            if (secureStorageAvailable) {
                putString(KEY_ACCESS_TOKEN, token)
                putString(KEY_REFRESH_TOKEN, refreshToken)
            }
            putString(KEY_USER_ROLE, role)
            putString(KEY_USER_NAME, name)
            putString(KEY_USER_EMAIL, email)
            putString(KEY_USER_AVATAR, avatarUrl)
            remove(KEY_VIEW_MODE)
            apply()
        }
        _authChanges.value = authGeneration
    }

    fun isRunnerView(): Boolean = prefs.getBoolean(KEY_VIEW_MODE, false)

    fun setRunnerView(runner: Boolean) {
        prefs.edit().putBoolean(KEY_VIEW_MODE, runner).apply()
    }

    // Token-only writer for the 401-refresh path. saveSession() would blank the
    // cached role/name/email/avatar (its params are required), and the refresh
    // response isn't a reliable place to re-derive them — a locally edited name
    // would be silently reverted. TokenAuthenticator calls this instead.
    @Synchronized
    fun updateTokens(accessToken: String, refreshToken: String?) {
        getAccessToken()?.let(accessTokens::add)
        while (accessTokens.size > 8) accessTokens.remove(accessTokens.first())
        if (!secureStorageAvailable) {
            memoryAccessToken = accessToken
            if (refreshToken != null) memoryRefreshToken = refreshToken
            return
        }
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
        return if (secureStorageAvailable) prefs.getString(KEY_ACCESS_TOKEN, null) else memoryAccessToken
    }

    fun getRefreshToken(): String? {
        return if (secureStorageAvailable) prefs.getString(KEY_REFRESH_TOKEN, null) else memoryRefreshToken
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

    @Synchronized
    fun clearSession() {
        authGeneration++
        accessTokens.clear()
        memoryAccessToken = null
        memoryRefreshToken = null
        // Before the two calls below, which reach outside this class: if either
        // throws, the credentials must already be off the disk.
        prefs.edit().clear().apply()
        _authChanges.value = authGeneration
        TetherWatchPrefs(appContext).clear()
        // A credential store reaching into service/ is the wrong shape, but a
        // signed-out phone must not keep ingesting to the old account and this
        // is the one call that stops it. A SessionEvents-driven stop would put
        // the responsibility back in the tether layer; not worth the churn yet.
        appContext.stopService(android.content.Intent(appContext, com.quickpitik.mobile.service.TetherIngestService::class.java))
    }
}
