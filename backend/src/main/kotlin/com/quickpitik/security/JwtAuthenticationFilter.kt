package com.quickpitik.security

import com.quickpitik.entity.Role
import io.jsonwebtoken.JwtException
import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.HttpHeaders
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter
import java.util.UUID

@Component
class JwtAuthenticationFilter(
    private val tokenProvider: JwtTokenProvider,
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
                val principal = AuthPrincipal(userId, email, role)
                val auth = UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
                auth.details = WebAuthenticationDetailsSource().buildDetails(request)
                SecurityContextHolder.getContext().authentication = auth
            } catch (_: JwtException) {
                // invalid / expired / malformed — leave unauthenticated
            } catch (_: IllegalArgumentException) {
                // malformed claims
            }
        }
        filterChain.doFilter(request, response)
    }

    private fun extractToken(request: HttpServletRequest): String? {
        val header = request.getHeader(HttpHeaders.AUTHORIZATION) ?: return null
        return if (header.startsWith("Bearer ", ignoreCase = true)) header.substring(7) else null
    }
}
