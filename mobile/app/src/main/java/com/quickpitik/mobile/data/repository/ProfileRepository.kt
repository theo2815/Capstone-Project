package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.SelfieRefDto
import com.quickpitik.mobile.data.remote.UserDto

interface ProfileRepository {
    suspend fun getSelfies(token: String): Result<List<SelfieRefDto>>
    
    suspend fun uploadSelfie(
        token: String,
        fileBytes: ByteArray,
        filename: String,
        contentType: String
    ): Result<SelfieRefDto>
    
    suspend fun deleteSelfie(token: String, selfieId: String): Result<Boolean>
    
    suspend fun setPrimarySelfie(token: String, selfieId: String): Result<List<SelfieRefDto>>
    
    suspend fun updateProfile(token: String, name: String): Result<UserDto>

    suspend fun uploadAvatar(
        token: String,
        fileBytes: ByteArray,
        filename: String,
        contentType: String
    ): Result<UserDto>

    /** Clears the avatar. Returns the updated user, whose avatarUrl is null. */
    suspend fun deleteAvatar(token: String): Result<UserDto>

    /**
     * [refreshToken] is this device's stored refresh token. The backend revokes
     * every OTHER session on a successful change and spares the one sent here.
     * Passing null revokes **everything, including this device** — the caller is
     * signed out of the phone it just changed the password on. Always pass
     * `SessionManager.getRefreshToken()`; null is only correct when there isn't
     * one (a session predating refresh-token persistence).
     */
    suspend fun changePassword(
        token: String,
        current: String,
        new: String,
        refreshToken: String?
    ): Result<String>

    /**
     * Step 1 of 2 of a sign-in-email change. Resolves with the backend's own
     * message on success — which says a link was SENT, not that the address
     * changed. Nothing moves until the link is redeemed from the new inbox
     * (web-only; see [com.quickpitik.mobile.data.remote.EmailChangeRequest]).
     */
    suspend fun requestEmailChange(
        token: String,
        newEmail: String,
        currentPassword: String
    ): Result<String>
}
