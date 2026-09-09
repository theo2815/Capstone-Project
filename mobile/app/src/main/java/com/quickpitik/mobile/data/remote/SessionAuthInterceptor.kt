package com.quickpitik.mobile.data.remote

import com.quickpitik.mobile.data.local.SessionManager
import okhttp3.Interceptor
import okhttp3.Response
import java.io.IOException

/** Bind authenticated requests to the login that supplied their bearer token. */
class SessionAuthInterceptor(private val session: SessionManager) : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val auth = request.header("Authorization") ?: return chain.proceed(request)
        if (!auth.startsWith("Bearer ")) return chain.proceed(request)
        val snapshot = session.authSnapshot(auth.removePrefix("Bearer "))
            ?: throw IOException("Account changed. Retry after signing in.")
        return chain.proceed(request.newBuilder().tag(SessionManager.AuthSession::class.java, snapshot).build())
    }
}
