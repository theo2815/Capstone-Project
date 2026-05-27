package com.quickpitik.mobile.data.remote

import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.Interceptor
import okhttp3.Response

// Rewrites the host on requests pointing at "localhost" or "127.0.0.1" to
// whatever host RetrofitClient.BASE_URL points at. The backend mints presigned
// storage URLs as http://localhost:8080/storage/... regardless of caller, and
// the Android emulator can't reach the host machine's localhost — only
// 10.0.2.2 resolves. Shared between the Coil ImageLoader (so AsyncImage
// paints) and the photo-download OkHttp client (so saves to gallery succeed).
object HostRewriteInterceptor : Interceptor {
    private val LOOPBACK = setOf("localhost", "127.0.0.1")

    override fun intercept(chain: Interceptor.Chain): Response {
        val req = chain.request()
        val url = req.url
        val backendHost = RetrofitClient.BASE_URL.toHttpUrlOrNull()?.host
        if (backendHost == null || url.host !in LOOPBACK || backendHost == url.host) {
            return chain.proceed(req)
        }
        val rewritten = url.newBuilder().host(backendHost).build()
        return chain.proceed(req.newBuilder().url(rewritten).build())
    }
}
