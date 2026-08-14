package com.quickpitik.mobile.data.remote

import android.content.Context
import com.quickpitik.mobile.data.local.SessionManager
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import com.google.gson.Gson
import retrofit2.HttpException
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
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
    const val BASE_URL = "http://192.168.1.232:8080/"

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

    // Two interceptors, one per verbosity. `bodyLogger` is what we want for JSON
    // traffic (auth, events, profile) — full request/response payloads in logcat
    // make debugging trivial. `headersLogger` is what we MUST use for multipart
    // photo uploads: at BODY level, OkHttp dumps every JPEG byte as a giant wall
    // of mojibake (1 MB photo = hundreds of unreadable log lines, plus the
    // logcat I/O itself measurably slows the upload). HEADERS keeps status code,
    // URL, and content-length without the binary spam.
    private val bodyLogger = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY
    }
    private val headersLogger = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.HEADERS
    }

    // Per-request router: pick HEADERS when the outgoing body is multipart
    // (i.e. the photo upload endpoint), BODY otherwise. Done at this layer so
    // we don't have to remember to silence logging at every call site that
    // streams binary — adding a new upload endpoint later "just works."
    private val loggingInterceptor = Interceptor { chain ->
        val request = chain.request()
        val isMultipart = request.body?.contentType()?.type == "multipart"
        val delegate = if (isMultipart) headersLogger else bodyLogger
        delegate.intercept(chain)
    }

    // Set once from QuickPitikApp.onCreate(), which always runs before any
    // screen, worker, or ViewModel can reach the network — so the lazy clients
    // below can safely assume it.
    private lateinit var appContext: Context

    fun init(context: Context) {
        appContext = context.applicationContext
    }

    // Default OkHttp read/write timeout is 10s — too tight for the PayMongo
    // Checkout Session call in dev (sandbox latency hits 12-20s easily). The
    // SocketTimeoutException retry storm froze the checkout sheet for ~20s
    // before bubbling up an error. 60s lets the gateway respond cleanly.
    private val okHttpClient by lazy {
        OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .authenticator(TokenAuthenticator(SessionManager.getInstance(appContext)))
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()
    }

    // Separate client for POST /auth/refresh, deliberately WITHOUT the
    // authenticator: the refresh call must never be able to trigger the refresh
    // path that issued it. TokenAuthenticator also guards on the path, so this
    // is belt-and-braces on the one call where a loop would be unrecoverable.
    private val refreshClient by lazy {
        OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build()
    }

    val apiService: QuickPitikApi by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(QuickPitikApi::class.java)
    }

    val refreshApi: QuickPitikApi by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(refreshClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(QuickPitikApi::class.java)
    }

    // Structured backend error (code + message) recovered from a failed call.
    // Retrofit throws HttpException for any non-2xx status (e.g. the backend's
    // 409 duplicate rejection) BEFORE the body is deserialized, so a caller that
    // needs the machine-readable error CODE (not just the message parseError
    // returns) must pull it off the thrown HttpException here. Returns null for
    // non-HTTP failures (timeouts, dropped connections) and unparseable bodies.
    // Reads the buffered errorBody ONCE — a caller must not also invoke
    // parseError on the same exception, or the second errorBody().string() hits
    // an already-drained buffer.
    fun parseHttpError(e: Throwable): ApiError? {
        if (e !is HttpException) return null
        return try {
            val errorBody = e.response()?.errorBody()?.string()
            Gson().fromJson(errorBody, ApiErrorEnvelope::class.java)?.errors?.firstOrNull()
        } catch (ex: Exception) {
            null
        }
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
        // Transport failures reach the UI as toasts, so they must not leak
        // "java.net.SocketTimeoutException" or "Failed to connect to
        // /192.168.1.232:8080". Runners on weak race-day Wi-Fi need to know it's
        // the connection, not the app. Order matters: the specific IOException
        // subtypes are checked before the generic catch-all below them.
        return when (e) {
            is UnknownHostException, is ConnectException ->
                "Couldn't reach QuickPitik — check your connection."
            is SocketTimeoutException ->
                "The connection timed out. Try again."
            is IOException ->
                "Network error. Check your connection and try again."
            else -> e.localizedMessage ?: "An unexpected error occurred"
        }
    }
}
