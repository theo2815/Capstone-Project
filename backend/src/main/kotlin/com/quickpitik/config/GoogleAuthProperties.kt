package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// Google sign-in ("Continue with Google") wiring for /auth/google. The one
// client ID is shared by every surface: the website's GIS button and the
// mobile Credential Manager both mint ID tokens whose `aud` is this WEB
// client ID, and this backend verifies against it. It is a public identifier
// (shipped in the website JS bundle), not a secret — env-driven anyway so
// dev/prod can point at different Google Cloud projects.
//
// `clientId`  — the Web application OAuth client ID from Google Cloud console.
//               Blank (the dev default) makes /auth/google answer 503
//               GOOGLE_AUTH_UNAVAILABLE instead of failing obscurely.
// `jwkSetUri` — Google's public signing keys; overridable only for tests.
@ConfigurationProperties(prefix = "app.auth.google")
data class GoogleAuthProperties(
    val clientId: String = "",
    val jwkSetUri: String = "https://www.googleapis.com/oauth2/v3/certs",
)
