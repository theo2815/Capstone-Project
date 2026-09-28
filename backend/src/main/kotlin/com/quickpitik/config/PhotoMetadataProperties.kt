package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

@ConfigurationProperties(prefix = "app.photo-metadata")
data class PhotoMetadataProperties(
    val recoveryEnabled: Boolean = true,
    val recoveryIntervalMs: Long = 60_000,
)
