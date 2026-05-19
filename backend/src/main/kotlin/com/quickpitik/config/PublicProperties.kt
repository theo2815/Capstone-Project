package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// Externally-reachable URLs the BE needs to embed in places its own request
// context can't tell it about — chiefly the receipt email, which is rendered
// from a background `@Async` listener with no inbound HTTP request.
//
// `apiBaseUrl` — absolute root of the public API (including `/api/v1`). Used
//                to build the bundle-download URL in the receipt email so
//                phones can tap the button and reach the BE. When running
//                ngrok for cross-device testing, set this AND
//                `app.storage.public-base-url` to the same tunnel root.
@ConfigurationProperties(prefix = "app.public")
data class PublicProperties(
    val apiBaseUrl: String = "http://localhost:8080/api/v1",
)
