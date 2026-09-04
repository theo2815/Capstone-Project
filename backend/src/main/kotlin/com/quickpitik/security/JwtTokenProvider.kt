package com.quickpitik.security

import com.quickpitik.config.JwtProperties
import com.quickpitik.entity.User
import io.jsonwebtoken.Claims
import io.jsonwebtoken.Jwts
import io.jsonwebtoken.security.Keys
import org.springframework.stereotype.Component
import java.nio.charset.StandardCharsets
import java.time.Instant
import java.util.Date
import javax.crypto.SecretKey

@Component
class JwtTokenProvider(private val props: JwtProperties) {
    private val key: SecretKey =
        Keys.hmacShaKeyFor(props.secret.toByteArray(StandardCharsets.UTF_8))

    fun createAccessToken(user: User): String {
        val now = Instant.now()
        val expiry = now.plus(props.accessTokenTtl)
        return Jwts.builder()
            .subject(user.id.toString())
            .claim("email", user.email)
            .claim("role", user.role.name)
            // Backstop, not the lockout mechanism. Both mint paths (login and
            // refresh) already refuse a suspended user, so today this is always
            // false at issue time; it exists so any future mint path that forgets
            // to check still produces a token JwtAuthenticationFilter rejects.
            // Real lockout = revoke-all-refresh-tokens on suspend + the refresh
            // gate, bounded by the 15-min access-token TTL. Closing that window
            // entirely would need a per-request DB read, which isn't worth it.
            .claim("suspended", user.suspendedAt != null)
            .issuedAt(Date.from(now))
            .expiration(Date.from(expiry))
            .signWith(key)
            .compact()
    }

    fun parse(token: String): Claims =
        Jwts.parser()
            .verifyWith(key)
            .build()
            .parseSignedClaims(token)
            .payload
}
