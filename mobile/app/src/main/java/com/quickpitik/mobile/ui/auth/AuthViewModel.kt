package com.quickpitik.mobile.ui.auth

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.ViewMode
import com.quickpitik.mobile.data.remote.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

sealed class AuthState {
    object Idle : AuthState()
    object Loading : AuthState()
    data class Success(val response: AuthResponse) : AuthState()
    data class Error(val message: String) : AuthState()
}

/**
 * Separate from [AuthState] because neither recovery call produces an
 * [AuthResponse] — forgot/reset return only a message and never establish a
 * session. Both screens share the one hoisted AuthViewModel, so each resets
 * this on entry (otherwise navigating forgot → reset would open already-Success).
 */
sealed class PasswordResetState {
    object Idle : PasswordResetState()
    object Loading : PasswordResetState()
    object Success : PasswordResetState()
    data class Error(val message: String) : PasswordResetState()
}

class AuthViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val _authState = MutableStateFlow<AuthState>(AuthState.Idle)
    val authState: StateFlow<AuthState> = _authState

    private val _passwordResetState = MutableStateFlow<PasswordResetState>(PasswordResetState.Idle)
    val passwordResetState: StateFlow<PasswordResetState> = _passwordResetState

    fun login(email: String, password: String) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            try {
                val envelope = RetrofitClient.apiService.login(LoginRequest(email, password))
                if (envelope.success && envelope.data != null) {
                    val authData = envelope.data
                    sessionManager.saveSession(
                        token = authData.accessToken,
                        role = authData.user.role,
                        name = authData.user.name,
                        email = authData.user.email,
                        avatarUrl = authData.user.avatarUrl,
                        refreshToken = authData.refreshToken,
                    )
                    _authState.value = AuthState.Success(authData)
                } else {
                    _authState.value = AuthState.Error(envelope.error ?: "Login failed. Please check credentials.")
                }
            } catch (e: Exception) {
                // RetrofitClient.parseError surfaces the backend envelope
                // message (e.g. the ACCOUNT_LOCKED "try again in N minutes"
                // copy) and maps transport failures to human copy — a login
                // timeout must not read "java.net.SocketTimeoutException".
                _authState.value = AuthState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun register(name: String, email: String, password: String, isPhotographer: Boolean) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            try {
                val role = if (isPhotographer) "PHOTOGRAPHER" else "RUNNER"
                val envelope = RetrofitClient.apiService.register(
                    RegisterRequest(name, email, password, role)
                )
                if (envelope.success && envelope.data != null) {
                    val authData = envelope.data
                    sessionManager.saveSession(
                        token = authData.accessToken,
                        role = authData.user.role,
                        name = authData.user.name,
                        email = authData.user.email,
                        avatarUrl = authData.user.avatarUrl,
                        refreshToken = authData.refreshToken,
                    )
                    _authState.value = AuthState.Success(authData)
                } else {
                    _authState.value = AuthState.Error(envelope.error ?: "Registration failed.")
                }
            } catch (e: Exception) {
                _authState.value = AuthState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    /**
     * POST /auth/forgot-password. The backend is deliberately anti-enumeration
     * silent — an unknown address still returns 200 with the same generic
     * message — so Success here means "the request was accepted", never
     * "this account exists". The screen's copy must not imply otherwise.
     */
    fun requestPasswordReset(email: String) {
        viewModelScope.launch {
            _passwordResetState.value = PasswordResetState.Loading
            try {
                val envelope = RetrofitClient.apiService.forgotPassword(
                    ForgotPasswordRequest(email.trim())
                )
                _passwordResetState.value = if (envelope.success) {
                    PasswordResetState.Success
                } else {
                    PasswordResetState.Error(
                        envelope.error ?: "Could not send reset link. Please try again."
                    )
                }
            } catch (e: Exception) {
                _passwordResetState.value = PasswordResetState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    /**
     * POST /auth/reset-password. Reset tokens are 15-minute, one-shot, so the
     * common failure here is an expired or already-used token — the screen's
     * fallback copy points the user back at requesting a fresh one. On success
     * the backend also revokes every refresh token for the account, logging
     * out other sessions.
     */
    fun confirmPasswordReset(token: String, newPassword: String) {
        viewModelScope.launch {
            _passwordResetState.value = PasswordResetState.Loading
            try {
                val envelope = RetrofitClient.apiService.resetPassword(
                    ResetPasswordRequest(token.trim(), newPassword)
                )
                _passwordResetState.value = if (envelope.success) {
                    PasswordResetState.Success
                } else {
                    PasswordResetState.Error(
                        envelope.error
                            ?: "Could not reset your password. The link may have expired — request a new one."
                    )
                }
            } catch (e: Exception) {
                _passwordResetState.value = PasswordResetState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun resetPasswordResetState() {
        _passwordResetState.value = PasswordResetState.Idle
    }

    fun resetState() {
        _authState.value = AuthState.Idle
    }

    /**
     * User-initiated sign-out.
     *
     * Until 2026-08-16 the sign-out buttons only called [resetState] and
     * navigated, so the cached JWT + refresh token survived in SharedPreferences.
     * Two consequences: the refresh token stayed usable server-side, and
     * `MainActivity`'s `startDestination` — which reads `getAccessToken()` —
     * put the next cold start straight back into the signed-in session.
     *
     * Order matters. The local session is cleared FIRST so sign-out is instant
     * and works offline; the revoke is fire-and-forget afterwards on the token
     * captured beforehand. A network failure must never strand the user signed
     * in, which is the same trade the website makes in `use-auth.ts`.
     */
    fun logout() {
        val refreshToken = sessionManager.getRefreshToken()
        sessionManager.clearSession()
        // clearSession() wipes the persisted runner-view flag; this resets the
        // in-memory one so the next login starts in the true role's home.
        ViewMode.reset(sessionManager)
        _authState.value = AuthState.Idle

        if (refreshToken == null) return
        viewModelScope.launch {
            runCatching { RetrofitClient.apiService.logout(LogoutRequest(refreshToken)) }
        }
    }
}
