package com.quickpitik.mobile.ui.auth

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.ViewMode
import com.quickpitik.mobile.data.remote.AuthResponse
import com.quickpitik.mobile.data.remote.ForgotPasswordRequest
import com.quickpitik.mobile.data.remote.GoogleLoginRequest
import com.quickpitik.mobile.data.remote.LoginRequest
import com.quickpitik.mobile.data.remote.LogoutRequest
import com.quickpitik.mobile.data.remote.RegisterRequest
import com.quickpitik.mobile.data.remote.ResetPasswordRequest
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.VerifyResetOtpRequest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

sealed class AuthState {
    object Idle : AuthState()
    object Loading : AuthState()
    data class Success(val response: AuthResponse) : AuthState()

    // Backend answered 422 ROLE_REQUIRED to a Google exchange: the account is
    // brand new and must pick RUNNER/PHOTOGRAPHER before it exists. The screen
    // shows the role sheet; completeGoogleSignup finishes the job.
    object GoogleRoleRequired : AuthState()
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

    // The one-shot continuation token from verify-reset-otp, consumed by
    // confirmPasswordReset. Memory only — a short-lived credential must never
    // touch SessionManager / SharedPreferences.
    private var resetToken: String? = null

    // The Google ID token parked while the role sheet is open, re-POSTed by
    // completeGoogleSignup. Same rule as resetToken: memory only.
    private var pendingGoogleIdToken: String? = null

    // Every path that mints a session (login / register / Google) lands here,
    // so SessionManager and the Success transition happen in exactly one place.
    private fun establishSession(authData: AuthResponse) {
        sessionManager.saveSession(
            token = authData.accessToken,
            role = authData.user.role,
            name = authData.user.name,
            email = authData.user.email,
            avatarUrl = authData.user.avatarUrl,
            refreshToken = authData.refreshToken,
        )
        _authState.value = AuthState.Success(authData)
    }

    fun login(email: String, password: String) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            try {
                val envelope = RetrofitClient.apiService.login(LoginRequest(email, password))
                if (envelope.success && envelope.data != null) {
                    establishSession(envelope.data)
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
                    establishSession(envelope.data)
                } else {
                    _authState.value = AuthState.Error(envelope.error ?: "Registration failed.")
                }
            } catch (e: Exception) {
                _authState.value = AuthState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    /**
     * POST /auth/google with the ID token minted by Credential Manager
     * (GoogleSignInRow). Success establishes a session exactly like login.
     * 422 ROLE_REQUIRED means this Google account is brand new — the token is
     * parked and the screen shows the role sheet, finished by
     * [completeGoogleSignup].
     */
    fun googleLogin(idToken: String) {
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            try {
                val envelope = RetrofitClient.apiService.googleLogin(GoogleLoginRequest(idToken))
                if (envelope.success && envelope.data != null) {
                    establishSession(envelope.data)
                } else {
                    _authState.value = AuthState.Error(envelope.error ?: "Google sign-in failed.")
                }
            } catch (e: Exception) {
                // parseHttpError drains the buffered errorBody, so parseError
                // must not re-read it — the fallback below is safe only
                // because parseError degrades to "HTTP <code>" on a drained
                // buffer and handles transport failures without touching it.
                val apiError = RetrofitClient.parseHttpError(e)
                if (apiError?.code == "ROLE_REQUIRED") {
                    pendingGoogleIdToken = idToken
                    _authState.value = AuthState.GoogleRoleRequired
                } else {
                    _authState.value = AuthState.Error(
                        apiError?.message ?: RetrofitClient.parseError(e)
                    )
                }
            }
        }
    }

    /**
     * Second leg for a brand-new Google account: re-POST the parked ID token
     * with the picked role. Google ID tokens live about an hour, so a stale
     * park just errors and the user taps the Google button again.
     */
    fun completeGoogleSignup(isPhotographer: Boolean) {
        val idToken = pendingGoogleIdToken
        if (idToken == null) {
            _authState.value = AuthState.Error("Google sign-in expired — try again.")
            return
        }
        viewModelScope.launch {
            _authState.value = AuthState.Loading
            try {
                val role = if (isPhotographer) "PHOTOGRAPHER" else "RUNNER"
                val envelope = RetrofitClient.apiService.googleLogin(
                    GoogleLoginRequest(idToken, role)
                )
                if (envelope.success && envelope.data != null) {
                    pendingGoogleIdToken = null
                    establishSession(envelope.data)
                } else {
                    _authState.value = AuthState.Error(envelope.error ?: "Google sign-in failed.")
                }
            } catch (e: Exception) {
                val apiError = RetrofitClient.parseHttpError(e)
                _authState.value = AuthState.Error(
                    apiError?.message ?: RetrofitClient.parseError(e)
                )
            }
        }
    }

    /** The user dismissed the role sheet — drop the parked token, back to Idle. */
    fun cancelGoogleSignup() {
        pendingGoogleIdToken = null
        _authState.value = AuthState.Idle
    }

    /**
     * Client-side sign-in failures (Credential Manager: no Google account on
     * the device, Play services missing) share the screens' error slot.
     */
    fun showError(message: String) {
        _authState.value = AuthState.Error(message)
    }

    /**
     * POST /auth/forgot-password. The backend is deliberately anti-enumeration
     * silent — an unknown address still returns 200 with the same generic
     * message — so Success here means "the request was accepted", never
     * "this account exists". The screen's copy must not imply otherwise.
     */
    fun requestPasswordReset(email: String) {
        // A new request restarts the flow — any continuation from an earlier
        // verify is dead server-side (invalidateOutstanding), so drop it here.
        resetToken = null
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
                        envelope.error ?: "Could not send a reset code. Please try again."
                    )
                }
            } catch (e: Exception) {
                _passwordResetState.value = PasswordResetState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    /**
     * POST /auth/verify-reset-otp. Trades the mailed 6-digit code for the
     * one-shot continuation token that [confirmPasswordReset] consumes. The
     * backend fails identically for an unknown email and a wrong code
     * (anti-enumeration), and kills the code after 5 wrong attempts.
     */
    fun verifyResetOtp(email: String, code: String) {
        viewModelScope.launch {
            _passwordResetState.value = PasswordResetState.Loading
            try {
                val envelope = RetrofitClient.apiService.verifyResetOtp(
                    VerifyResetOtpRequest(email.trim(), code)
                )
                if (envelope.success && envelope.data != null) {
                    resetToken = envelope.data.resetToken
                    _passwordResetState.value = PasswordResetState.Success
                } else {
                    _passwordResetState.value = PasswordResetState.Error(
                        envelope.error
                            ?: "That code didn't work. It may have expired — resend a new one."
                    )
                }
            } catch (e: Exception) {
                _passwordResetState.value = PasswordResetState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    /**
     * POST /auth/reset-password with the continuation token from [verifyResetOtp]
     * (15-minute, one-shot). On success the backend also revokes every refresh
     * token for the account, logging out other sessions.
     */
    fun confirmPasswordReset(newPassword: String) {
        val token = resetToken
        if (token == null) {
            _passwordResetState.value =
                PasswordResetState.Error("Your verification expired — request a new code.")
            return
        }
        viewModelScope.launch {
            _passwordResetState.value = PasswordResetState.Loading
            try {
                val envelope = RetrofitClient.apiService.resetPassword(
                    ResetPasswordRequest(token, newPassword)
                )
                if (envelope.success) {
                    resetToken = null
                    _passwordResetState.value = PasswordResetState.Success
                } else {
                    _passwordResetState.value = PasswordResetState.Error(
                        envelope.error
                            ?: "Could not reset your password. Your verification may have expired — start over."
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
