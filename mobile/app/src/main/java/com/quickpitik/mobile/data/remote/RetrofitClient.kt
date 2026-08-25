package com.quickpitik.mobile.data.remote

import android.content.Context
import com.quickpitik.mobile.BuildConfig
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
    // Compiled-in fallback. Used verbatim on a fresh install, in every release
    // build, and whenever the debug override is cleared.
    //
    // When using the Android Studio Emulator, "http://10.0.2.2:8080/" routes to your PC's backend.
    // If you use a physical phone via USB instead of Wi-Fi, run "adb reverse tcp:8080 tcp:8080".
    const val DEFAULT_BASE_URL = "http://192.168.1.9:8080/"

    // The live backend origin. Was a `const val` until 2026-08-16, which meant
    // every laptop IP change cost a Kotlin edit + recompile + reinstall — and on
    // a camera shoot the phone's USB-C port is occupied by the body, so there is
    // no cable to push that build over. Debug builds can now set it in-app (see
    // setBaseUrl); release builds are pinned to DEFAULT_BASE_URL.
    //
    // Still read-only to the outside world, and still named BASE_URL, so every
    // existing reader — HostRewriteInterceptor, the five photographer screens,
    // the getters below — is unchanged.
    @Volatile
    private var _baseUrl: String = DEFAULT_BASE_URL

    val BASE_URL: String
        get() = _baseUrl

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

    // WebSocket origin ("ws://host:port"), derived from BASE_URL for the same
    // reason as the getters above — one host change flows everywhere. The BE
    // registers its handlers at /ws/... on the server ROOT, not under the REST
    // /api/v1 prefix, so callers pass an absolute path and we contribute only
    // the scheme + authority. Mirrors the website's buildWsUrl().
    val wsOrigin: String
        get() {
            val url = BASE_URL.toHttpUrlOrNull() ?: return "ws://10.0.2.2:8080"
            val scheme = if (url.isHttps) "wss" else "ws"
            return "$scheme://${url.host}:${url.port}"
        }

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
        // Deliberately NOT SessionManager's prefs file: the server address has
        // to survive sign-out, and clearSession() wiping it mid-test would be a
        // genuinely nasty bug. Runs here — before the first network call — so
        // the cached clients below are built against the right origin.
        if (BuildConfig.DEBUG) {
            readPersistedBaseUrl()?.let { _baseUrl = it }
        }
    }

    private fun prefs() =
        appContext.getSharedPreferences(DEV_PREF_NAME, Context.MODE_PRIVATE)

    private fun readPersistedBaseUrl(): String? =
        prefs().getString(KEY_BASE_URL, null)?.let(::normalizeBaseUrl)

    /**
     * Points the app at a different backend. Debug builds only — a release
     * build ignores this and stays on [DEFAULT_BASE_URL], so no shipped APK can
     * be talked into calling an arbitrary host.
     *
     * Returns false (changing nothing) if [raw] isn't a usable HTTP(S) origin.
     * On success the next call rebuilds Retrofit, so no app restart is needed —
     * but an ALREADY-OPEN WebSocket keeps the host it dialled. That's fine for
     * the only caller: the field lives on Login, before any socket is opened.
     */
    fun setBaseUrl(raw: String): Boolean {
        if (!BuildConfig.DEBUG) return false
        val normalized = normalizeBaseUrl(raw) ?: return false
        _baseUrl = normalized
        prefs().edit().putString(KEY_BASE_URL, normalized).apply()
        return true
    }

    /** Drops the override and returns to the compiled-in [DEFAULT_BASE_URL]. */
    fun resetBaseUrl() {
        if (!BuildConfig.DEBUG) return
        _baseUrl = DEFAULT_BASE_URL
        prefs().edit().remove(KEY_BASE_URL).apply()
    }

    /**
     * "192.168.1.5:8080" → "http://192.168.1.5:8080/". Forgiving about the two
     * things a human typing an IP on a phone keyboard always omits — the scheme
     * and the trailing slash (Retrofit throws on a baseUrl without one) — and
     * strict about everything else: null means don't touch the current value.
     */
    private fun normalizeBaseUrl(raw: String): String? {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return null
        val withScheme =
            if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) trimmed
            else "http://$trimmed"
        val withSlash = if (withScheme.endsWith("/")) withScheme else "$withScheme/"
        // toHttpUrlOrNull is the same parser Retrofit and OkHttp use, so if it
        // accepts the value every downstream consumer will too.
        return if (withSlash.toHttpUrlOrNull() != null) withSlash else null
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

    // Retrofit bakes the baseUrl in at build time, so these two can't be plain
    // `by lazy` any more — a URL change after first use would be silently
    // ignored. Each is cached against the URL that built it and rebuilt only on
    // a mismatch, which is the ONLY thing that changes: the OkHttp clients above
    // are URL-independent, so the connection pool, timeouts and authenticator
    // all survive a server switch untouched.
    @Volatile
    private var apiCache: Pair<String, QuickPitikApi>? = null

    @Volatile
    private var refreshCache: Pair<String, QuickPitikApi>? = null

    val apiService: QuickPitikApi
        get() {
            val url = _baseUrl
            apiCache?.let { (cachedUrl, api) -> if (cachedUrl == url) return api }
            return synchronized(this) {
                apiCache?.let { (cachedUrl, api) -> if (cachedUrl == url) return api }
                buildApi(url, okHttpClient).also { apiCache = url to it }
            }
        }

    val refreshApi: QuickPitikApi
        get() {
            val url = _baseUrl
            refreshCache?.let { (cachedUrl, api) -> if (cachedUrl == url) return api }
            return synchronized(this) {
                refreshCache?.let { (cachedUrl, api) -> if (cachedUrl == url) return api }
                buildApi(url, refreshClient).also { refreshCache = url to it }
            }
        }

    private fun buildApi(url: String, client: OkHttpClient): QuickPitikApi =
        Retrofit.Builder()
            .baseUrl(url)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(QuickPitikApi::class.java)

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

    private const val DEV_PREF_NAME = "quickpitik_dev"
    private const val KEY_BASE_URL = "dev_base_url"
}
