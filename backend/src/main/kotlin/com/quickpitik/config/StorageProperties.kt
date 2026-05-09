package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.storage")
data class StorageProperties(
    val backend: Backend = Backend.LOCAL,
    val bucket: String = "quickpitik-dev",
    val region: String = "ap-southeast-1",
    val endpoint: String? = null,
    val accessKey: String? = null,
    val secretKey: String? = null,
    val pathStyleAccess: Boolean = false,
    val localRoot: String = ".storage",
    val publicBaseUrl: String? = null,
    val presignedTtl: PresignedTtl = PresignedTtl(),
) {
    enum class Backend { LOCAL, S3 }

    data class PresignedTtl(
        val thumbnail: Duration = Duration.ofHours(1),
        val avatar: Duration = Duration.ofHours(1),
        val selfie: Duration = Duration.ofHours(1),
        val cover: Duration = Duration.ofHours(1),
        val watermark: Duration = Duration.ofHours(1),
        val runnerDownload: Duration = Duration.ofMinutes(15),
        val photographerDownload: Duration = Duration.ofMinutes(5),
    )
}
