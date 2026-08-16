package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

class ProfileRepositoryImpl : ProfileRepository {
    private val api = RetrofitClient.apiService

    override suspend fun getSelfies(token: String): Result<List<SelfieRefDto>> {
        return try {
            val response = api.getSelfies("Bearer $token")
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to fetch selfies"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun uploadSelfie(
        token: String,
        fileBytes: ByteArray,
        filename: String,
        contentType: String
    ): Result<SelfieRefDto> {
        return try {
            val mediaType = contentType.toMediaTypeOrNull() ?: "image/jpeg".toMediaTypeOrNull()
            val requestBody = fileBytes.toRequestBody(mediaType)
            val part = MultipartBody.Part.createFormData("file", filename, requestBody)
            
            val response = api.uploadSelfie("Bearer $token", part)
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to upload selfie"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun deleteSelfie(token: String, selfieId: String): Result<Boolean> {
        return try {
            val response = api.deleteSelfie("Bearer $token", selfieId)
            if (response.success && response.data != null) {
                Result.success(response.data.removed)
            } else {
                Result.failure(Exception(response.error ?: "Failed to delete selfie"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun setPrimarySelfie(token: String, selfieId: String): Result<List<SelfieRefDto>> {
        return try {
            val response = api.setPrimarySelfie("Bearer $token", selfieId)
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to set primary selfie"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun updateProfile(token: String, name: String): Result<UserDto> {
        return try {
            val response = api.updateProfile("Bearer $token", ProfileUpdateRequest(name))
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to update profile name"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun uploadAvatar(
        token: String,
        fileBytes: ByteArray,
        filename: String,
        contentType: String
    ): Result<UserDto> {
        return try {
            val mediaType = contentType.toMediaTypeOrNull() ?: "image/jpeg".toMediaTypeOrNull()
            val requestBody = fileBytes.toRequestBody(mediaType)
            val part = MultipartBody.Part.createFormData("file", filename, requestBody)

            val response = api.uploadAvatar("Bearer $token", part)
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to upload avatar"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun changePassword(
        token: String,
        current: String,
        new: String,
        refreshToken: String?
    ): Result<String> {
        return try {
            val response = api.changePassword(
                "Bearer $token",
                PasswordChangeRequest(current, new, refreshToken)
            )
            if (response.success && response.data != null) {
                Result.success(response.data["message"] ?: "Password changed successfully")
            } else {
                Result.failure(Exception(response.error ?: "Failed to change password"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun deleteAvatar(token: String): Result<UserDto> {
        return try {
            val response = api.deleteAvatar("Bearer $token")
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to remove photo"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun requestEmailChange(
        token: String,
        newEmail: String,
        currentPassword: String
    ): Result<String> {
        return try {
            val response = api.requestEmailChange(
                "Bearer $token",
                EmailChangeRequest(newEmail, currentPassword)
            )
            if (response.success && response.data != null) {
                // Prefer the server's wording — it is careful to say a link was
                // sent rather than that the address changed.
                Result.success(
                    response.data["message"]
                        ?: "Check your new inbox for the confirmation link."
                )
            } else {
                Result.failure(Exception(response.error ?: "Failed to request the email change"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }
}
