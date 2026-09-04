package com.quickpitik.mobile.data.remote

import com.quickpitik.mobile.data.local.SessionEvents
import com.quickpitik.mobile.data.local.SessionManager
import okhttp3.Authenticator
import okhttp3.Request
import okhttp3.Response
import okhttp3.Route

/**
 * Recovers the session when the 15-minute access-token TTL expires mid-use.
 * Without this, F7's stored refresh token had no consumer: the user was force
 * logged out every 15 minutes — which also caps a tethered shoot, since the
 * upload worker's requests start 401ing (see the 2026-08-14 live-auto-upload
 * journal, "Known limitations").
 *
 * An Authenticator (not an Interceptor) because OkHttp only invokes it on an
 * actual 401, so the happy path costs nothing, and it hands us the retry.
 */
class TokenAuthenticator(
    private val sessionManager: SessionManager,
) : Authenticator {

    override fun authenticate(route: Route?, response: Response): Request? {
        // Never touch /auth/*. Those endpoints answer 401 for ordinary reasons —
        // a wrong password on login, a rotated-away refresh token — none of which
        // mean "the session expired." Refreshing there is meaningless, and
        // clearing the session would wipe the login screen mid-typing. Bailing on
        // /auth/refresh specifically is also what makes the recursion terminate.
        if (response.request.url.encodedPath.contains(AUTH_PATH)) return null

        // One retry per request: if the server 401s even a freshly minted token
        // we must stop rather than spin.
        if (responseCount(response) >= 2) return null

        val failedAuthHeader = response.request.header("Authorization")

        // Serialised so a burst of parallel 401s (the upload worker runs 3
        // concurrent uploads) performs ONE refresh, not three. The backend
        // rotates refresh tokens, so racing refreshes would invalidate here.
        synchronized(this) {
            val currentToken = sessionManager.getAccessToken()

            // Another thread already refreshed while this request was in flight.
            // Retry with what's in prefs instead of spending the refresh token.
            if (currentToken != null && "Bearer $currentToken" != failedAuthHeader) {
                return retryWith(response.request, currentToken)
            }

            val refreshToken = sessionManager.getRefreshToken() ?: return giveUp()

            // Blocking execute() is correct here: authenticate() is already off
            // the main thread, and refreshApi is a client with NO authenticator
            // so this call can never re-enter the path that issued it.
            val refreshed = runCatching {
                RetrofitClient.refreshApi
                    .refreshToken(RefreshRequest(refreshToken))
                    .execute()
            }.getOrNull()

            // Distinguish "the server rejected the refresh token" from "the call
            // never completed." A dropped connection must NOT log the user out —
            // on race-day Wi-Fi that would happen constantly. Here we just let
            // the original 401 surface and keep the session for the next try.
            if (refreshed == null) return null

            val auth = refreshed.body()?.data
            // A real rejection (401/403 on a valid-looking token) means the
            // session is genuinely dead — that's the only case worth clearing.
            // The rejection's error body distinguishes WHY: an ACCOUNT_SUSPENDED
            // refresh must land on login with the reason shown, not read like an
            // ordinary expiry.
            if (!refreshed.isSuccessful || auth == null) {
                val err = runCatching {
                    com.google.gson.Gson().fromJson(
                        refreshed.errorBody()?.string(),
                        ApiErrorEnvelope::class.java,
                    )?.errors?.firstOrNull()
                }.getOrNull()
                val reason = err?.message?.takeIf { err.code == "ACCOUNT_SUSPENDED" }
                return giveUp(reason)
            }

            sessionManager.updateTokens(auth.accessToken, auth.refreshToken)
            return retryWith(response.request, auth.accessToken)
        }
    }

    private fun giveUp(reason: String? = null): Request? {
        sessionManager.clearSession()
        SessionEvents.raiseForcedLogout(reason)
        return null
    }

    private fun retryWith(request: Request, token: String): Request =
        request.newBuilder()
            .header("Authorization", "Bearer $token")
            .build()

    private fun responseCount(response: Response): Int {
        var count = 1
        var prior = response.priorResponse
        while (prior != null) {
            count++
            prior = prior.priorResponse
        }
        return count
    }

    private companion object {
        const val AUTH_PATH = "/auth/"
    }
}
