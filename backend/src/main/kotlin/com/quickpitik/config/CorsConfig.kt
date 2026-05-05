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
            allowedHeaders = listOf("*")
            exposedHeaders = listOf(HttpHeaders.AUTHORIZATION)
            allowCredentials = true
            maxAge = 3600
        }
        val source = UrlBasedCorsConfigurationSource()
        source.registerCorsConfiguration("/**", config)
        return source
    }

    private object HttpHeaders {
        const val AUTHORIZATION = "Authorization"
    }
}
