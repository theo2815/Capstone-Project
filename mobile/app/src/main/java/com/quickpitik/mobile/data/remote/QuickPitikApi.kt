package com.quickpitik.mobile.data.remote

import retrofit2.http.Body
import retrofit2.http.POST

interface QuickPitikApi {
    @POST("api/v1/auth/login")
    suspend fun login(@Body request: LoginRequest): ApiResponseEnvelope<AuthResponse>

    @POST("api/v1/auth/register")
    suspend fun register(@Body request: RegisterRequest): ApiResponseEnvelope<AuthResponse>
}
