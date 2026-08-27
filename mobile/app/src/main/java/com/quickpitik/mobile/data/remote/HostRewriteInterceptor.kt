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
    override fun intercept(chain: Interceptor.Chain): Response {
        val req = chain.request()
        val backend = RetrofitClient.BASE_URL.toHttpUrlOrNull() ?: return chain.proceed(req)
        val rewritten = rewriteLoopbackUrl(req.url, backend)
        return chain.proceed(if (rewritten == req.url) req else req.newBuilder().url(rewritten).build())
    }
}
