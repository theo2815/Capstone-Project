package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.EmailChangeRequest
import com.quickpitik.mobile.data.remote.PasswordChangeRequest
import com.quickpitik.mobile.data.remote.ProfileUpdateRequest
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.SelfieRefDto
import com.quickpitik.mobile.data.remote.UserDto
import com.quickpitik.mobile.data.remote.apiCall
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

class ProfileRepositoryImpl : ProfileRepository {
    private val api get() = RetrofitClient.apiService

    override suspend fun getSelfies(token: String): Result<List<SelfieRefDto>> =
        apiCall("Failed to fetch selfies") { api.getSelfies("Bearer $token") }

    override suspend fun uploadSelfie(
        token: String,
        fileBytes: ByteArray,
        filename: String,
        contentType: String
    ): Result<SelfieRefDto> =
        apiCall("Failed to upload selfie") {
            api.uploadSelfie("Bearer $token", filePart(fileBytes, filename, contentType))
        }

    override suspend fun deleteSelfie(token: String, selfieId: String): Result<Boolean> =
        apiCall("Failed to delete selfie") { api.deleteSelfie("Bearer $token", selfieId) }
            .map { it.removed }

    override suspend fun setPrimarySelfie(token: String, selfieId: String): Result<List<SelfieRefDto>> =
        apiCall("Failed to set primary selfie") { api.setPrimarySelfie("Bearer $token", selfieId) }

    override suspend fun updateProfile(token: String, name: String): Result<UserDto> =
        apiCall("Failed to update profile name") {
            api.updateProfile("Bearer $token", ProfileUpdateRequest(name))
        }

    override suspend fun uploadAvatar(
        token: String,
        fileBytes: ByteArray,
        filename: String,
        contentType: String
    ): Result<UserDto> =
        apiCall("Failed to upload avatar") {
            api.uploadAvatar("Bearer $token", filePart(fileBytes, filename, contentType))
        }

    override suspend fun changePassword(
        token: String,
        current: String,
        new: String,
        refreshToken: String?
    ): Result<String> =
        apiCall("Failed to change password") {
            api.changePassword("Bearer $token", PasswordChangeRequest(current, new, refreshToken))
        }.map { it["message"] ?: "Password changed successfully" }

    override suspend fun deleteAvatar(token: String): Result<UserDto> =
        apiCall("Failed to remove photo") { api.deleteAvatar("Bearer $token") }

    override suspend fun requestEmailChange(
        token: String,
        newEmail: String,
        currentPassword: String
    ): Result<String> =
        apiCall("Failed to request the email change") {
            api.requestEmailChange("Bearer $token", EmailChangeRequest(newEmail, currentPassword))
        }
            // Prefer the server's wording — it is careful to say a link was
            // sent rather than that the address changed.
            .map { it["message"] ?: "Check your new inbox for the confirmation link." }

    private fun filePart(bytes: ByteArray, filename: String, contentType: String): MultipartBody.Part {
        val mediaType = contentType.toMediaTypeOrNull() ?: "image/jpeg".toMediaTypeOrNull()
        return MultipartBody.Part.createFormData("file", filename, bytes.toRequestBody(mediaType))
    }
}
