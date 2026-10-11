package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "jwt")
data class JwtProperties(
    val secret: String,
    val accessTokenTtl: Duration,
    val refreshTokenTtl: Duration,
    // How long a just-rotated refresh token is still accepted, for a
    // client whose rotation answer was lost (RefreshTokenService). Default
    // keeps the ten test constructions compiling.
    val refreshRotationGrace: Duration = Duration.ofMinutes(10),
)
