package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.net.URI
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
    val directUpload: DirectUpload = DirectUpload(),
) {
    init {
        require(directUpload.putTtl > Duration.ZERO) { "directUpload.putTtl must be positive" }
        require(directUpload.intentTtl > directUpload.putTtl) {
            "directUpload.intentTtl must be greater than putTtl"
        }
        require(directUpload.maxBytes in 1..MAX_UPLOAD_BYTES) {
            "directUpload.maxBytes must be between 1 and $MAX_UPLOAD_BYTES"
        }
        require(directUpload.maxPendingPerUser > 0) { "directUpload.maxPendingPerUser must be positive" }
        require(directUpload.maxPendingBytesPerUser >= directUpload.maxBytes) {
            "directUpload.maxPendingBytesPerUser must allow at least one maximum-size upload"
        }
        require(directUpload.verificationConcurrency > 0) {
            "directUpload.verificationConcurrency must be positive"
        }
        require(directUpload.verificationLease > DIRECT_OPERATION_BUDGET) {
            "directUpload.verificationLease must exceed the direct-upload operation budget"
        }
        require(directUpload.cleanupDelay > Duration.ZERO) { "directUpload.cleanupDelay must be positive" }
        require(directUpload.cleanupInterval > Duration.ZERO) { "directUpload.cleanupInterval must be positive" }
        require(directUpload.cleanupBatchSize > 0) { "directUpload.cleanupBatchSize must be positive" }
        require(directUpload.cleanupRecheckWindow > Duration.ZERO) {
            "directUpload.cleanupRecheckWindow must be positive"
        }
        require(!directUpload.websiteEnabled || directUpload.enabled) {
            "directUpload.websiteEnabled requires directUpload.enabled"
        }
        if (directUpload.enabled) {
            require(backend == Backend.S3 && isR2Endpoint()) {
                "direct upload requires an S3 backend with a Cloudflare R2 endpoint"
            }
        }
    }

    enum class Backend { LOCAL, S3 }

    fun isR2Endpoint(): Boolean = endpoint
        ?.takeIf { it.isNotBlank() }
        ?.let { URI.create(it) }
        ?.let { uri ->
            uri.scheme.equals("https", ignoreCase = true) &&
                uri.host?.endsWith(".r2.cloudflarestorage.com", ignoreCase = true) == true
        } == true

    data class DirectUpload(
        val enabled: Boolean = false,
        val websiteEnabled: Boolean = false,
        val putTtl: Duration = Duration.ofMinutes(15),
        val intentTtl: Duration = Duration.ofHours(24),
        val maxBytes: Long = MAX_UPLOAD_BYTES,
        val maxPendingPerUser: Int = 32,
        val maxPendingBytesPerUser: Long = 800L * 1024 * 1024,
        val verificationConcurrency: Int = 2,
        val verificationLease: Duration = Duration.ofMinutes(5),
        val cleanupDelay: Duration = Duration.ofHours(1),
        val cleanupInterval: Duration = Duration.ofMinutes(1),
        val cleanupBatchSize: Int = 100,
        val cleanupRecheckWindow: Duration = Duration.ofDays(7),
    )

    data class PresignedTtl(
        val thumbnail: Duration = Duration.ofHours(1),
        val avatar: Duration = Duration.ofHours(1),
        val selfie: Duration = Duration.ofHours(1),
        val cover: Duration = Duration.ofHours(1),
        val watermark: Duration = Duration.ofHours(1),
        val runnerDownload: Duration = Duration.ofMinutes(15),
        val photographerDownload: Duration = Duration.ofMinutes(5),
    )

    private companion object {
        const val MAX_UPLOAD_BYTES: Long = 25L * 1024 * 1024
        val DIRECT_OPERATION_BUDGET: Duration = Duration.ofSeconds(120)
    }
}
