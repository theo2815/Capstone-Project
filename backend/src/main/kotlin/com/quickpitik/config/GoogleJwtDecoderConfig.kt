package com.quickpitik.config

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.security.oauth2.core.DelegatingOAuth2TokenValidator
import org.springframework.security.oauth2.core.OAuth2Error
import org.springframework.security.oauth2.core.OAuth2TokenValidator
import org.springframework.security.oauth2.core.OAuth2TokenValidatorResult
import org.springframework.security.oauth2.jwt.Jwt
import org.springframework.security.oauth2.jwt.JwtDecoder
import org.springframework.security.oauth2.jwt.JwtTimestampValidator
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder

// Verifier for Google-signed ID tokens (RS256 against Google's JWKS). Only
// GoogleAuthService consumes this bean — our own access tokens stay on the
// jjwt HS256 path (JwtTokenProvider), and nothing in the filter chain reads
// a JwtDecoder. Nimbus fetches + caches the key set on first decode, not at
// boot, so the app still starts offline.
@Configuration
class GoogleJwtDecoderConfig {
    @Bean
    fun googleJwtDecoder(properties: GoogleAuthProperties): JwtDecoder {
        val decoder = NimbusJwtDecoder.withJwkSetUri(properties.jwkSetUri).build()
        // setJwtValidator REPLACES the default validator, so the timestamp
        // check must be re-added explicitly alongside ours.
        decoder.setJwtValidator(
            DelegatingOAuth2TokenValidator(
                JwtTimestampValidator(),
                googleClaimsValidator(properties),
            ),
        )
        return decoder
    }

    // iss is read as a raw string, not jwt.issuer: Google historically signs
    // with both issuer forms, and the bare "accounts.google.com" would fail
    // the URL conversion jwt.issuer performs. A validation failure surfaces
    // as JwtValidationException (a BadJwtException) — GoogleAuthService maps
    // that to 401 INVALID_GOOGLE_TOKEN.
    private fun googleClaimsValidator(properties: GoogleAuthProperties) =
        OAuth2TokenValidator<Jwt> { jwt ->
            val issuerOk = jwt.getClaimAsString("iss") in GOOGLE_ISSUERS
            val audienceOk = properties.clientId.isNotBlank() &&
                jwt.audience.contains(properties.clientId)
            if (issuerOk && audienceOk) {
                OAuth2TokenValidatorResult.success()
            } else {
                OAuth2TokenValidatorResult.failure(
                    OAuth2Error("invalid_token", "Not a Google token for this app", null),
                )
            }
        }

    private companion object {
        val GOOGLE_ISSUERS = setOf("https://accounts.google.com", "accounts.google.com")
    }
}
