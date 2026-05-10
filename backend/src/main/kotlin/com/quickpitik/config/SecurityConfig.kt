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
            .authorizeHttpRequests { auth ->
                auth.requestMatchers(
                    "/api/v1/auth/register",
                    "/api/v1/auth/login",
                    "/api/v1/auth/refresh",
                    "/api/v1/auth/forgot-password",
                    "/api/v1/auth/reset-password",
                ).permitAll()
                auth.requestMatchers("/error", "/actuator/**").permitAll()
                auth.requestMatchers("/ws/**").permitAll()
                auth.requestMatchers(HttpMethod.GET, "/api/v1/events", "/api/v1/events/**").permitAll()
                auth.requestMatchers(HttpMethod.GET, "/api/v1/regions").permitAll()
                auth.requestMatchers(HttpMethod.GET, "/api/v1/platform/**").permitAll()
                auth.requestMatchers(HttpMethod.GET, "/api/v1/public/**").permitAll()
                auth.requestMatchers(HttpMethod.POST, "/api/v1/events/*/photos/search-by-face").authenticated()
                auth.requestMatchers(HttpMethod.POST, "/api/v1/orders").permitAll()
                auth.requestMatchers(HttpMethod.POST, "/api/v1/payments/webhook/**").permitAll()
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
}
