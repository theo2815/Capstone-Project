package com.quickpitik.mobile.ui.runner

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
import com.quickpitik.mobile.data.readAtMost
import com.quickpitik.mobile.data.remote.SelfieRefDto
import com.quickpitik.mobile.data.repository.ProfileRepository
import com.quickpitik.mobile.data.repository.ProfileRepositoryImpl
import com.quickpitik.mobile.ui.auth.validateEmail
import com.quickpitik.mobile.ui.auth.validateNewPassword
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ProfileViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val repository: ProfileRepository = ProfileRepositoryImpl()

    private val _selfiesState = MutableStateFlow<List<SelfieRefDto>>(emptyList())
    val selfiesState: StateFlow<List<SelfieRefDto>> = _selfiesState.asStateFlow()

    private val _selfiesLoading = MutableStateFlow(false)
    val selfiesLoading: StateFlow<Boolean> = _selfiesLoading.asStateFlow()

    private val _selfiesError = MutableStateFlow<String?>(null)
    val selfiesError: StateFlow<String?> = _selfiesError.asStateFlow()

    private val _profileName = MutableStateFlow(sessionManager.getUserName() ?: "")
    val profileName: StateFlow<String> = _profileName.asStateFlow()

    private val _profileEmail = MutableStateFlow(sessionManager.getUserEmail() ?: "")
    val profileEmail: StateFlow<String> = _profileEmail.asStateFlow()

    private val _nameUpdateSuccess = MutableStateFlow(false)
    val nameUpdateSuccess: StateFlow<Boolean> = _nameUpdateSuccess.asStateFlow()

    private val _nameUpdateError = MutableStateFlow<String?>(null)
    val nameUpdateError: StateFlow<String?> = _nameUpdateError.asStateFlow()

    private val _passwordUpdateSuccess = MutableStateFlow(false)
    val passwordUpdateSuccess: StateFlow<Boolean> = _passwordUpdateSuccess.asStateFlow()

    private val _passwordUpdateError = MutableStateFlow<String?>(null)
    val passwordUpdateError: StateFlow<String?> = _passwordUpdateError.asStateFlow()

    // Step 1 of 2 of a sign-in-email change. `emailChangeMessage` holds the
    // backend's confirmation copy ("we sent a link"), NOT a changed address —
    // `profileEmail` deliberately stays put, because the swap only happens when
    // the link is redeemed from the new inbox.
    private val _emailChangeSubmitting = MutableStateFlow(false)
    val emailChangeSubmitting: StateFlow<Boolean> = _emailChangeSubmitting.asStateFlow()

    private val _emailChangeMessage = MutableStateFlow<String?>(null)
    val emailChangeMessage: StateFlow<String?> = _emailChangeMessage.asStateFlow()

    private val _emailChangeError = MutableStateFlow<String?>(null)
    val emailChangeError: StateFlow<String?> = _emailChangeError.asStateFlow()

    // False when there was no stored refresh token to spare, so the backend
    // revoked every session including this one. The access token keeps working
    // for <=15 min and then the device is bounced — the success copy has to
    // admit that instead of claiming a clean change.
    private val _passwordSessionKept = MutableStateFlow(true)
    val passwordSessionKept: StateFlow<Boolean> = _passwordSessionKept.asStateFlow()

    private val _avatarUrl = MutableStateFlow(sessionManager.getAvatarUrl())
    val avatarUrl: StateFlow<String?> = _avatarUrl.asStateFlow()

    private val _avatarUploading = MutableStateFlow(false)
    val avatarUploading: StateFlow<Boolean> = _avatarUploading.asStateFlow()

    private val _avatarError = MutableStateFlow<String?>(null)
    val avatarError: StateFlow<String?> = _avatarError.asStateFlow()

    fun fetchSelfies() {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            _selfiesLoading.value = true
            _selfiesError.value = null
            repository.getSelfies(token)
                .onSuccess { list ->
                    _selfiesState.value = list
                }
                .onFailure { err ->
                    _selfiesError.value = err.message ?: "Failed to retrieve selfies"
                }
            _selfiesLoading.value = false
        }
    }

    fun uploadSelfie(uri: Uri) {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            _selfiesLoading.value = true
            _selfiesError.value = null
            try {
                val contentResolver = getApplication<Application>().contentResolver
                val bytes = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { it.readAtMost(MAX_UPLOAD_BYTES + 1) }
                }
                if (bytes != null) {
                    if (bytes.size > MAX_UPLOAD_BYTES) {
                        _selfiesError.value = "Selfies must be 8 MB or smaller"
                        return@launch
                    }
                    
                    val mimeType = contentResolver.getType(uri) ?: "image/jpeg"
                    val filename = "selfie_${System.currentTimeMillis()}.jpg"
                    
                    repository.uploadSelfie(token, bytes, filename, mimeType)
                        .onSuccess {
                            fetchSelfies()
                        }
                        .onFailure { err ->
                            _selfiesError.value = err.message ?: "Upload failed"
                        }
                } else {
                    _selfiesError.value = "Unable to read image file"
                }
            } catch (e: Exception) {
                _selfiesError.value = e.message ?: "An unexpected error occurred during image reading"
            } finally {
                _selfiesLoading.value = false
            }
        }
    }

    fun deleteSelfie(selfieId: String) {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            _selfiesLoading.value = true
            _selfiesError.value = null
            repository.deleteSelfie(token, selfieId)
                .onSuccess {
                    fetchSelfies()
                }
                .onFailure { err ->
                    _selfiesError.value = err.message ?: "Failed to delete selfie"
                }
            _selfiesLoading.value = false
        }
    }

    fun setPrimarySelfie(selfieId: String) {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            _selfiesLoading.value = true
            _selfiesError.value = null
            repository.setPrimarySelfie(token, selfieId)
                .onSuccess { list ->
                    _selfiesState.value = list
                }
                .onFailure { err ->
                    _selfiesError.value = err.message ?: "Failed to set primary selfie"
                }
            _selfiesLoading.value = false
        }
    }

    fun updateName(name: String) {
        val token = sessionManager.getAccessToken() ?: return
        if (name.trim().isEmpty()) {
            _nameUpdateError.value = "Name cannot be empty"
            return
        }
        viewModelScope.launch {
            _nameUpdateError.value = null
            _nameUpdateSuccess.value = false
            repository.updateProfile(token, name.trim())
                .onSuccess { userDto ->
                    sessionManager.saveUserName(userDto.name)
                    _profileName.value = userDto.name
                    _nameUpdateSuccess.value = true
                }
                .onFailure { err ->
                    _nameUpdateError.value = err.message ?: "Failed to update profile name"
                }
        }
    }

    fun resetNameState() {
        _nameUpdateSuccess.value = false
        _nameUpdateError.value = null
    }

    fun changePassword(current: String, new: String) {
        val token = sessionManager.getAccessToken() ?: return
        if (current.isEmpty() || new.isEmpty()) {
            _passwordUpdateError.value = "Passwords cannot be empty"
            return
        }
        // Shared with the auth screens (and a verbatim port of the website's
        // rule) so the floor and the 72-byte bcrypt ceiling are stated once.
        validateNewPassword(new)?.let {
            _passwordUpdateError.value = it
            return
        }
        // Spares THIS device when revoking other sessions. Null only when the
        // session predates refresh-token persistence — the backend then revokes
        // everything, so say so rather than showing a plain success message.
        val refreshToken = sessionManager.getRefreshToken()
        viewModelScope.launch {
            _passwordUpdateError.value = null
            _passwordUpdateSuccess.value = false
            repository.changePassword(token, current, new, refreshToken)
                .onSuccess {
                    _passwordSessionKept.value = refreshToken != null
                    _passwordUpdateSuccess.value = true
                }
                .onFailure { err ->
                    _passwordUpdateError.value = err.message ?: "Failed to update password"
                }
        }
    }

    fun resetPasswordState() {
        _passwordUpdateSuccess.value = false
        _passwordUpdateError.value = null
    }

    fun requestEmailChange(newEmail: String, currentPassword: String) {
        val token = sessionManager.getAccessToken() ?: return
        // Same validator the auth screens use, so the rule and the copy match
        // what a runner already saw at register.
        validateEmail(newEmail)?.let {
            _emailChangeError.value = it
            return
        }
        if (currentPassword.isEmpty()) {
            _emailChangeError.value = "Enter your current password to confirm."
            return
        }
        if (newEmail.trim().equals(sessionManager.getUserEmail(), ignoreCase = true)) {
            _emailChangeError.value = "That's already your sign-in email."
            return
        }
        viewModelScope.launch {
            _emailChangeSubmitting.value = true
            _emailChangeError.value = null
            _emailChangeMessage.value = null
            repository.requestEmailChange(token, newEmail.trim(), currentPassword)
                .onSuccess { _emailChangeMessage.value = it }
                .onFailure { err ->
                    _emailChangeError.value = err.message ?: "Failed to request the email change"
                }
            _emailChangeSubmitting.value = false
        }
    }

    fun resetEmailChangeState() {
        _emailChangeMessage.value = null
        _emailChangeError.value = null
    }

    fun removeAvatar() {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            _avatarUploading.value = true
            _avatarError.value = null
            repository.deleteAvatar(token)
                .onSuccess { userDto ->
                    // Mirror uploadAvatar: SessionManager is what every other
                    // surface reads the avatar from, so it has to be cleared
                    // too or the dashboard keeps rendering the old one.
                    sessionManager.saveAvatarUrl(userDto.avatarUrl)
                    _avatarUrl.value = userDto.avatarUrl
                }
                .onFailure { err ->
                    _avatarError.value = err.message ?: "Failed to remove photo"
                }
            _avatarUploading.value = false
        }
    }

    fun uploadAvatar(uri: Uri) {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            _avatarUploading.value = true
            _avatarError.value = null
            try {
                val contentResolver = getApplication<Application>().contentResolver
                val bytes = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { it.readAtMost(MAX_UPLOAD_BYTES + 1) }
                }
                if (bytes != null) {
                    if (bytes.size > MAX_UPLOAD_BYTES) {
                        _avatarError.value = "Profile photos must be 8 MB or smaller"
                        return@launch
                    }

                    val mimeType = contentResolver.getType(uri) ?: "image/jpeg"
                    val filename = "avatar_${System.currentTimeMillis()}.jpg"

                    repository.uploadAvatar(token, bytes, filename, mimeType)
                        .onSuccess { userDto ->
                            sessionManager.saveAvatarUrl(userDto.avatarUrl)
                            _avatarUrl.value = userDto.avatarUrl
                        }
                        .onFailure { err ->
                            _avatarError.value = err.message ?: "Avatar upload failed"
                        }
                } else {
                    _avatarError.value = "Unable to read image file"
                }
            } catch (e: Exception) {
                _avatarError.value = e.message ?: "An unexpected error occurred during image reading"
            } finally {
                _avatarUploading.value = false
            }
        }
    }

}
