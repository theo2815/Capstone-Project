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

    suspend fun changePassword(
        token: String,
        current: String,
        new: String
    ): Result<String>
}
