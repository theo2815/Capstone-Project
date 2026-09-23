package com.quickpitik.security

import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.web.filter.OncePerRequestFilter
import java.security.MessageDigest

class OriginSecretFilter(secret: String) : OncePerRequestFilter() {
    private val expected = secret.toByteArray(Charsets.UTF_8)

    override fun doFilterInternal(request: HttpServletRequest, response: HttpServletResponse, chain: FilterChain) {
        if (request.servletPath != "/actuator/health" &&
            !MessageDigest.isEqual(expected, request.getHeader("X-Origin-Secret")?.toByteArray(Charsets.UTF_8))
        ) {
            response.status = HttpServletResponse.SC_FORBIDDEN
            return
        }
        chain.doFilter(request, response)
    }
}
