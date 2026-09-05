package com.quickpitik.config

import com.quickpitik.security.JsonAccessDeniedHandler
import com.quickpitik.security.JsonAuthenticationEntryPoint
import com.quickpitik.security.JwtAuthenticationFilter
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.http.HttpMethod
import org.springframework.security.authentication.AuthenticationManager
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity
import org.springframework.security.config.annotation.web.builders.HttpSecurity
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity
import org.springframework.security.config.http.SessionCreationPolicy
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.security.web.SecurityFilterChain
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter
import org.springframework.security.web.header.writers.ReferrerPolicyHeaderWriter
import org.springframework.security.web.util.matcher.RegexRequestMatcher
import org.springframework.web.cors.CorsConfigurationSource

@Configuration
@EnableWebSecurity
@EnableMethodSecurity
class SecurityConfig(
    private val jwtAuthenticationFilter: JwtAuthenticationFilter,
    private val corsConfigurationSource: CorsConfigurationSource,
    private val authenticationEntryPoint: JsonAuthenticationEntryPoint,
    private val accessDeniedHandler: JsonAccessDeniedHandler,
) {
    @Bean
    fun passwordEncoder(): PasswordEncoder = BCryptPasswordEncoder(12)

    @Bean
    fun authenticationManager(config: AuthenticationConfiguration): AuthenticationManager =
        config.authenticationManager

    @Bean
    fun securityFilterChain(http: HttpSecurity): SecurityFilterChain {
        http
            .csrf { it.disable() }
            .cors { it.configurationSource(corsConfigurationSource) }
            .sessionManagement { it.sessionCreationPolicy(SessionCreationPolicy.STATELESS) }
            .headers { headers ->
                // Spring Security defaults (nosniff, X-Frame-Options DENY,
                // HSTS-on-secure-request) stay; these two the defaults omit.
                // HSTS only fires once the request LOOKS secure — behind a
                // TLS-terminating proxy configure Tomcat's RemoteIpValve
                // (server.tomcat.remoteip.*, see application.yml) so it does.
                // CSP intentionally absent: JSON API; springdoc serves its own UI.
                headers.referrerPolicy {
                    it.policy(ReferrerPolicyHeaderWriter.ReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN)
                }
                headers.permissionsPolicyHeader { it.policy("camera=(), microphone=(), geolocation=()") }
            }
            .authorizeHttpRequests { auth ->
                auth.requestMatchers(
                    "/api/v1/auth/register",
                    "/api/v1/auth/login",
                    // Google ID-token exchange — the token in the body is the
                    // credential, verified against Google's JWKS.
                    "/api/v1/auth/google",
                    "/api/v1/auth/refresh",
                    "/api/v1/auth/logout",
                    "/api/v1/auth/forgot-password",
                    "/api/v1/auth/verify-reset-otp",
                    "/api/v1/auth/reset-password",
                    // Opened from the NEW inbox, so the browser usually has no
                    // session. The opaque token in the body is the credential.
                    "/api/v1/auth/confirm-email-change",
                    // Same shape: followed from a mail client, token is the
                    // credential. Note /auth/resend-verification is NOT here —
                    // its caller is always signed in, so it falls through to
                    // anyRequest().authenticated() below.
                    "/api/v1/auth/verify-email",
                ).permitAll()
                auth.requestMatchers("/error").permitAll()
                // Actuator: health is a public liveness probe; everything else
                // (metrics, info, …) is ADMIN-only. The previous blanket
                // permitAll("/actuator/**") predated the actuator dependency —
                // an armed hole for whoever added it, which 2026-08-27 did.
                auth.requestMatchers("/actuator/health").permitAll()
                auth.requestMatchers("/actuator/**").hasRole("ADMIN")
                // Generated API docs. Whether they exist at all is decided by
                // springdoc's own `enabled` flags (API_DOCS_ENABLED) — with
                // those off there is no handler here to reach, so this rule
                // stops mattering rather than becoming a hole.
                auth.requestMatchers(
                    "/v3/api-docs",
                    "/v3/api-docs/**",
                    "/swagger-ui.html",
                    "/swagger-ui/**",
                ).permitAll()
                auth.requestMatchers("/ws/**").permitAll()
                // LocalFs storage mount (StaticResourceConfig) — <img> tags
                // can't carry the bearer token; in prod S3StorageService serves
                // the same content from a different origin and this rule is a
                // no-op (no handler registered).
                auth.requestMatchers(HttpMethod.GET, "/storage/**").permitAll()
                auth.requestMatchers(HttpMethod.GET, "/api/v1/events", "/api/v1/events/**").permitAll()
                auth.requestMatchers(HttpMethod.GET, "/api/v1/regions").permitAll()
                auth.requestMatchers(HttpMethod.GET, "/api/v1/platform/**").permitAll()
                auth.requestMatchers(HttpMethod.GET, "/api/v1/public/**").permitAll()
                // Photo verification is public by design — whoever holds a
                // screenshot may ask whose photo it is. IP-keyed rate bucket
                // + MIME/size gate in the controller; the answer is
                // attribution only, never a photo URL.
                auth.requestMatchers(HttpMethod.POST, "/api/v1/public/photos/verify").permitAll()
                // Face search is a guest surface — a visitor finds their photos
                // first and signs up at checkout, if at all. The multipart
                // variant is written for it (nullable principal, IP-keyed rate
                // bucket, 5 MB + MIME whitelist). The stored-selfie JSON
                // variant sharing this path enforces its own 401 in the
                // controller, so a signed-out caller can never reach another
                // user's selfie. Same shape as the token-gated order rules
                // below: permitted here, authorized in the service layer.
                auth.requestMatchers(HttpMethod.POST, "/api/v1/events/*/photos/search-by-face").permitAll()
                auth.requestMatchers(HttpMethod.POST, "/api/v1/orders").permitAll()
                // Coupon preview is a guest checkout surface too — the code is
                // printed on public photo cards. IP-keyed bucket in the
                // controller; the answer is a priced item list, never a URL.
                auth.requestMatchers(HttpMethod.POST, "/api/v1/coupons/preview").permitAll()
                auth.requestMatchers(
                    HttpMethod.GET,
                    "/api/v1/orders/mobile-return",
                    "/api/v1/orders/mobile-cancel",
                ).permitAll()
                // Guest order status polling. Service-layer token check enforces auth.
                auth.requestMatchers(guestOrderGet("/status")).permitAll()
                // Guest cancel of a live QR — same token gate, in the service layer.
                auth.requestMatchers(guestOrderRoute(HttpMethod.POST, "/cancel")).permitAll()
                // Bundle download (token-gated; works for both runners + guests
                // because a top-level navigation can't carry the JWT).
                auth.requestMatchers(guestOrderGet("/download-bundle")).permitAll()
                // Guest order detail (token-gated, anti-IDOR via service layer).
                // The UUID regex keeps future named sibling routes private.
                auth.requestMatchers(guestOrderGet("")).permitAll()
                auth.requestMatchers(HttpMethod.POST, "/api/v1/payments/webhook/**").permitAll()
                // Internal ai-api job webhook (Phase C). Authorization is the
                // HMAC verifier, same pattern as the payment webhook above.
                auth.requestMatchers(HttpMethod.POST, "/api/v1/internal/ai-webhooks").permitAll()
                auth.requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                auth.anyRequest().authenticated()
            }
            .exceptionHandling { ex ->
                ex.authenticationEntryPoint(authenticationEntryPoint)
                ex.accessDeniedHandler(accessDeniedHandler)
            }
            .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter::class.java)

        return http.build()
    }

    internal companion object {
        private const val UUID_PATH =
            "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89aAbB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
        private const val GUEST_ORDER_PATH = "^/api/v1/orders/$UUID_PATH"

        // Matcher for a token-gated guest order route (detail = "", plus
        // "/status", "/download-bundle"). `internal` so SecurityConfigGuestOrderTest
        // can exercise the exact production pattern against a real query string.
        //
        // RegexRequestMatcher matches against `servletPath + '?' + queryString`,
        // NOT the path alone — so the anchor goes after the PATH, then an optional
        // query string. Every guest route is reached with ?token=..., and a bare
        // `...$` here rejected the real request, 401'd the guest, and bounced web
        // to /login (mobile hung). `(?:\?.*)?` keeps sibling paths private while
        // permitting the token param.
        internal fun guestOrderGet(suffix: String): RegexRequestMatcher = guestOrderRoute(HttpMethod.GET, suffix)

        internal fun guestOrderRoute(method: HttpMethod, suffix: String): RegexRequestMatcher =
            RegexRequestMatcher.regexMatcher(method, "$GUEST_ORDER_PATH$suffix(?:\\?.*)?\$")
    }
}
