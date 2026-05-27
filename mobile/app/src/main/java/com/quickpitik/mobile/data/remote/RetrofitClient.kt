package com.quickpitik.mobile.data.remote

import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import com.google.gson.Gson
import retrofit2.HttpException
import java.util.concurrent.TimeUnit

object RetrofitClient {
    // DEMO TODO (2026-05-28): if running on a physical phone over Wi-Fi, set this
    // to your laptop's Wi-Fi IPv4 (e.g. "http://192.168.X.Y:8080/"). Run
    // `ipconfig` in PowerShell while connected to the demo Wi-Fi to find it.
    // All image URL rewrites (avatar/cover/banner/share/profile/settings) read
    // from this constant via `backendHost` / `backendOrigin` below, so changing
    // this single line is enough — no need to hunt down per-screen hardcodes.
    //
    // When using the Android Studio Emulator, "http://10.0.2.2:8080/" routes to your PC's backend.
    // If you use a physical phone via USB instead of Wi-Fi, run "adb reverse tcp:8080 tcp:8080".
    const val BASE_URL = "http://10.0.2.2:8080/"

    // Single source of truth for image URL rewriting across the photographer
    // screens. Derived from BASE_URL so any host change (emulator → Wi-Fi IP →
    // ngrok URL) automatically flows to every avatar/cover/banner site without
    // a per-screen edit. Falls back to "10.0.2.2" if BASE_URL is malformed.
    val backendHost: String
        get() = BASE_URL.toHttpUrlOrNull()?.host ?: "10.0.2.2"

    // BASE_URL without the trailing slash — for path-based image URLs that
    // start with "/" (e.g. presigned storage paths from the backend).
    val backendOrigin: String
        get() = BASE_URL.trimEnd('/')

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }

    // Default OkHttp read/write timeout is 10s — too tight for the PayMongo
    // Checkout Session call in dev (sandbox latency hits 12-20s easily). The
    // SocketTimeoutException retry storm froze the checkout sheet for ~20s
    // before bubbling up an error. 60s lets the gateway respond cleanly.
    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
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
