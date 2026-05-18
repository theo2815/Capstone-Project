package com.quickpitik.mobile.data.remote

import okhttp3.MultipartBody
import retrofit2.http.*

interface QuickPitikApi {
    @POST("api/v1/auth/login")
    suspend fun login(@Body request: LoginRequest): ApiResponseEnvelope<AuthResponse>

    @POST("api/v1/auth/register")
    suspend fun register(@Body request: RegisterRequest): ApiResponseEnvelope<AuthResponse>

    @Multipart
    @POST("api/v1/me/photographer/events/{eventId}/photos")
    suspend fun uploadPhoto(
        @Header("Authorization") token: String,
        @Path("eventId") eventId: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<UploadedPhotoDto>

    @GET("api/v1/me/photographer/events")
    suspend fun getPhotographerEvents(
        @Header("Authorization") token: String,
        @Query("withUploads") withUploads: Boolean = false
    ): ApiResponseEnvelope<PaginatedResponse<PhotographerEventSummaryDto>>
}
