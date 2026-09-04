package com.quickpitik.config

import org.springframework.beans.factory.annotation.Value
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.web.cors.CorsConfiguration
import org.springframework.web.cors.CorsConfigurationSource
import org.springframework.web.cors.UrlBasedCorsConfigurationSource

@Configuration
class CorsConfig(
    @Value("\${app.cors.allowed-origins}") private val allowedOriginsCsv: String,
) {
    @Bean
    fun corsConfigurationSource(): CorsConfigurationSource {
        val origins = allowedOriginsCsv.split(",").map { it.trim() }.filter { it.isNotBlank() }
        val config = CorsConfiguration().apply {
            allowedOrigins = origins
            allowedMethods = listOf("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
            // Explicit whitelist rather than "*": the CORS spec forbids the
            // wildcard alongside allowCredentials=true, so browsers were only
            // honouring it by grace. These four are the complete set the
            // website + mobile clients actually send.
            allowedHeaders = listOf(
                "Content-Type",
                HttpHeaders.AUTHORIZATION,
                "Accept",
                "Idempotency-Key",
            )
            // Retry-After: every 429 (rate limit + lockout) carries it, but
            // browser JS can only read headers listed here. Content-Disposition:
            // lets a fetch()-based download read the server-chosen filename.
            // X-Total-Count: the message inboxes keep a bare-array body (mobile
            // parity) and put the true row total here for the web inbox to page.
            exposedHeaders = listOf(
                HttpHeaders.AUTHORIZATION,
                HttpHeaders.RETRY_AFTER,
                HttpHeaders.CONTENT_DISPOSITION,
                X_TOTAL_COUNT,
            )
            allowCredentials = true
            maxAge = 3600
        }
        val source = UrlBasedCorsConfigurationSource()
        source.registerCorsConfiguration("/**", config)
        return source
    }

    private object HttpHeaders {
        const val AUTHORIZATION = "Authorization"
        const val RETRY_AFTER = "Retry-After"
        const val CONTENT_DISPOSITION = "Content-Disposition"
    }

    private companion object {
        const val X_TOTAL_COUNT = "X-Total-Count"
    }
}
