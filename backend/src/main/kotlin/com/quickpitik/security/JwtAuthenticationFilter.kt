package com.quickpitik.security

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ApiError
import com.quickpitik.common.ApiResponse
import com.quickpitik.entity.Role
import io.jsonwebtoken.JwtException
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter
import java.util.UUID

@Component
class JwtAuthenticationFilter(
    private val tokenProvider: JwtTokenProvider,
    private val objectMapper: ObjectMapper,
) : OncePerRequestFilter() {

    override fun doFilterInternal(
        request: HttpServletRequest,
        response: HttpServletResponse,
        filterChain: FilterChain,
    ) {
        val token = extractToken(request)
        if (token != null && SecurityContextHolder.getContext().authentication == null) {
            try {
                val claims = tokenProvider.parse(token)
                val userId = UUID.fromString(claims.subject)
                val email = claims["email"] as? String ?: error("missing email claim")
                val role = Role.valueOf(claims["role"] as? String ?: error("missing role claim"))
                // Absent claim = token minted before the suspension gate landed;
                // treat as active so in-flight sessions aren't mass-logged-out.
                val suspended = claims["suspended"] as? Boolean ?: false
                if (!suspended) {
                    val principal = AuthPrincipal(userId, email, role)
                    val auth = UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
                    auth.details = WebAuthenticationDetailsSource().buildDetails(request)
                    SecurityContextHolder.getContext().authentication = auth
                }
                // Suspended: leave unauthenticated so JsonAuthenticationEntryPoint
                // returns the standard 401 envelope. The client's refresh attempt
                // then fails (revoked token, or ACCOUNT_SUSPENDED from
                // AuthService.refresh), which is what ends the session.
            } catch (ex: Exception) {
                if (ex is JwtException || ex is IllegalArgumentException) {
                    response.contentType = MediaType.APPLICATION_JSON_VALUE
                    response.status = HttpServletResponse.SC_UNAUTHORIZED
                    objectMapper.writeValue(
                        response.outputStream,
                        ApiResponse.failure(ApiError(code = "UNAUTHORIZED", message = "Invalid or expired token")),
                    )
                    return
                }
            }
        }
        filterChain.doFilter(request, response)
    }

    private fun extractToken(request: HttpServletRequest): String? {
        val header = request.getHeader(HttpHeaders.AUTHORIZATION) ?: return null
        return if (header.startsWith("Bearer ", ignoreCase = true)) header.substring(7) else null
    }
}

