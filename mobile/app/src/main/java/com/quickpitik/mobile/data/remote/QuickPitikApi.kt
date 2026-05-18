package com.quickpitik.mobile.data.remote

import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Body

interface QuickPitikApi {
    @POST("/v1/auth/login")
    suspend fun login(@Body request: Map<String, String>): Map<String, String>

    @POST("/v1/events/{eventId}/photos/upload-init")
    suspend fun initUpload(
        @Path("eventId") eventId: String,
        @Body metadata: Map<String, String>
    ): Map<String, String> // Returns signed S3 URL and upload UUID

    @POST("/v1/events/{eventId}/photos/finalize")
    suspend fun finalizeUpload(
        @Path("eventId") eventId: String,
        @Body data: Map<String, String>
    ): Map<String, String>
}
