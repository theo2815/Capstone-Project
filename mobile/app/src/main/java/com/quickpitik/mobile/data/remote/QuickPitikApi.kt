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

    @GET("api/v1/events")
    suspend fun getPublicEvents(
        @Query("status") status: String = "ACTIVE",
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<EventDto>>

    @GET("api/v1/events/{slug}/photos")
    suspend fun getEventPhotos(
        @Path("slug") slug: String,
        @Query("bib") bib: String? = null,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<PhotoDto>>

    @Multipart
    @POST("api/v1/events/{slug}/photos/search-by-face")
    suspend fun searchPhotosByFace(
        @Path("slug") slug: String,
        @Part selfie: MultipartBody.Part,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<PhotoDto>>
}
