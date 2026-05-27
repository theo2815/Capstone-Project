package com.quickpitik.mobile.ui.auth

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import retrofit2.HttpException

sealed class AuthState {
    object Idle : AuthState()
    object Loading : AuthState()
    data class Success(val response: AuthResponse) : AuthState()
    data class Error(val message: String) : AuthState()
}

class AuthViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val _authState = MutableStateFlow<AuthState>(AuthState.Idle)
    val authState: StateFlow<AuthState> = _authState

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
            } catch (e: HttpException) {
                _authState.value = AuthState.Error(parseHttpError(e))
            } catch (e: Exception) {
                _authState.value = AuthState.Error(e.localizedMessage ?: "Login failed. Please try again.")
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
            } catch (e: HttpException) {
                _authState.value = AuthState.Error(parseHttpError(e))
            } catch (e: Exception) {
                _authState.value = AuthState.Error(e.localizedMessage ?: "Registration failed. Please try again.")
            }
        }
    }

    private fun parseHttpError(e: HttpException): String {
        return try {
            val errorBody = e.response()?.errorBody()?.string()
            val parsedError = Gson().fromJson(errorBody, ApiErrorEnvelope::class.java)
            parsedError?.errors?.firstOrNull()?.message ?: "Operation failed. Please try again."
        } catch (ex: Exception) {
            "Connection failed. Please try again."
        }
    }

    fun resetState() {
        _authState.value = AuthState.Idle
    }
}
