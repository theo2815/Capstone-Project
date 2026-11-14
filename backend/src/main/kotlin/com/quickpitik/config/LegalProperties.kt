package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// The Privacy Policy / Terms version stamped on new sign-ups that tick the box.
// A plain value in application.yml, not an env var: it changes with the text,
// in code. The website's LEGAL_VERSION must match it (website version test).
@ConfigurationProperties(prefix = "app.legal")
data class LegalProperties(
    val currentVersion: String,
    // RV-L4 (P5): photographers who accepted a version below this are asked to accept the current Terms again.
    // Set in code with the legal 1.7 texts (RV-L7). Unset means nobody is asked, and nobody counts as having
    // consented to automatic hides (RV-2.2a).
    val photographerMinVersion: String? = null,
) {
    // An empty yml value binds as "", not null: read the minimum only through this.
    fun minVersionOrNull(): String? = photographerMinVersion?.takeIf { it.isNotBlank() }
}
