package com.quickpitik.mobile.ui.runner

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.remote.PhotoVerifyResultDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

sealed class VerifyState {
    object Idle : VerifyState()
    object Uploading : VerifyState()
    data class Result(val result: PhotoVerifyResultDto) : VerifyState()
    data class Error(val message: String) : VerifyState()
}

// "Whose photo is this?" (V42) — public reverse attribution. Uploads one image
// to POST /public/photos/verify (no auth) and shows who shot it. Nothing is
// stored server-side; the response carries attribution only, never a photo URL.
class VerifyViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow<VerifyState>(VerifyState.Idle)
    val state: StateFlow<VerifyState> = _state

    fun verify(uri: Uri) {
        viewModelScope.launch {
            _state.value = VerifyState.Uploading
            val cr = getApplication<Application>().contentResolver
            val bytes = try {
                cr.openInputStream(uri)?.use { it.readBytes() }
            } catch (e: Exception) {
                null
            }
            if (bytes == null) {
                _state.value = VerifyState.Error("Couldn't read that image. Pick another.")
                return@launch
            }
            // Backend cap is 10 MB — reject before the upload so a big file fails
            // fast with a clear message instead of a wasted 413 round-trip.
            if (bytes.size > MAX_BYTES) {
                _state.value = VerifyState.Error("That image is over 10 MB. Pick a smaller one.")
                return@launch
            }
            try {
                // Send the real mime so the backend's jpeg/png gate sees the true
                // type (a webp/heic pick is rejected server-side with a clear code).
                val mime = cr.getType(uri) ?: "image/jpeg"
                val filename = if (mime.contains("png", ignoreCase = true)) "photo.png" else "photo.jpg"
                val body = bytes.toRequestBody(mime.toMediaTypeOrNull(), 0, bytes.size)
                val part = MultipartBody.Part.createFormData("file", filename, body)
                val response = RetrofitClient.apiService.verifyPhoto(part)
                if (response.success && response.data != null) {
                    _state.value = VerifyState.Result(response.data)
                } else {
                    _state.value = VerifyState.Error(response.error ?: "Couldn't check that photo. Try again.")
                }
            } catch (e: Exception) {
                _state.value = VerifyState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun reset() { _state.value = VerifyState.Idle }

    private companion object {
        const val MAX_BYTES = 10 * 1024 * 1024
    }
}
