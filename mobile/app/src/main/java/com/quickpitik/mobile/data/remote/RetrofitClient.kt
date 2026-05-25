package com.quickpitik.mobile.data.remote

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import com.google.gson.Gson
import retrofit2.HttpException

object RetrofitClient {
    // When using the Android Studio Emulator, use "http://10.0.2.2:8080/" to route requests to your PC's backend.
    // If you switch to a physical phone via USB, run "adb reverse tcp:8080 tcp:8080" in the terminal.
    private const val BASE_URL = "http://localhost:8080/"

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .build()

    val apiService: QuickPitikApi by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(QuickPitikApi::class.java)
    }

    fun parseError(e: Throwable): String {
        if (e is HttpException) {
            return try {
                val errorBody = e.response()?.errorBody()?.string()
                val parsedError = Gson().fromJson(errorBody, ApiErrorEnvelope::class.java)
                parsedError?.errors?.firstOrNull()?.message ?: "HTTP ${e.code()}"
            } catch (ex: Exception) {
                "HTTP ${e.code()}"
            }
        }
        return e.localizedMessage ?: "An unexpected error occurred"
    }
}
