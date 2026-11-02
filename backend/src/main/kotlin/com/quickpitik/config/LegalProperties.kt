package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// The Privacy Policy / Terms version stamped on new sign-ups that tick the box.
// A plain value in application.yml, not an env var: it changes with the text,
// in code. The website's LEGAL_VERSION must match it (website version test).
@ConfigurationProperties(prefix = "app.legal")
data class LegalProperties(
    val currentVersion: String,
)
