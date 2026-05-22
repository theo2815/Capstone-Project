package com.quickpitik.mobile.ui.runner

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.SelfieRefDto
import com.quickpitik.mobile.data.repository.ProfileRepository
import com.quickpitik.mobile.data.repository.ProfileRepositoryImpl
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

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

    init {
        fetchSelfies()
    }

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
                val inputStream = contentResolver.openInputStream(uri)
                if (inputStream != null) {
                    val bytes = inputStream.readBytes()
                    inputStream.close()
                    
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
        if (new.length < 8) {
            _passwordUpdateError.value = "New password must be at least 8 characters"
            return
        }
        viewModelScope.launch {
            _passwordUpdateError.value = null
            _passwordUpdateSuccess.value = false
            repository.changePassword(token, current, new)
                .onSuccess {
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
}
