package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// `backfillEnabled` — runs PhotoGridBackfill, which gives photos published
//                     before the 400 px grid tile their grid.jpg. Off by
//                     default; the owner switches it on per environment and
//                     off again after its "thumbnail backfill complete" line.
@ConfigurationProperties(prefix = "app.thumbnails")
data class ThumbnailProperties(
    val backfillEnabled: Boolean = false,
)
