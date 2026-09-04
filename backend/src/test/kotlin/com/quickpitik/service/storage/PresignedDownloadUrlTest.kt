package com.quickpitik.service.storage

import com.quickpitik.config.StorageProperties
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.springframework.mock.env.MockEnvironment
import java.nio.file.Path
import java.time.Duration

// presignedDownloadUrl contract: the disposition must reach the client as part
// of the URL the SERVER minted — LocalFs via the filter's query params, S3/R2
// via a SIGNED response-content-disposition (an unsigned appended param breaks
// the SigV4 signature and 403s, which is the bug this method exists to avoid).
class PresignedDownloadUrlTest {

    @TempDir
    lateinit var tempDir: Path

    @Test
    fun `localfs download url carries the params the disposition filter consumes`() {
        val service = LocalFsStorageService(
            StorageProperties(localRoot = tempDir.toString()),
            MockEnvironment(),
        )
        val url = service.presignedDownloadUrl(
            "events/e1/photo.jpg",
            Duration.ofMinutes(15),
            "quickpitik-bib-03-107.jpg",
        )
        assertTrue(url.contains("disposition=attachment"))
        assertTrue(url.contains("filename=quickpitik-bib-03-107.jpg"))
        assertTrue(url.contains("expires="))
    }

    @Test
    fun `s3 download url signs response-content-disposition into the query`() {
        s3Service().use { service ->
            val url = service.presignedDownloadUrl(
                "events/e1/photo.jpg",
                Duration.ofMinutes(15),
                "quickpitik-bib-42.jpg",
            )
            assertTrue(url.contains("response-content-disposition="))
            assertTrue(url.contains("X-Amz-Signature="), "disposition must be inside the signed request")
        }
    }

    @Test
    fun `plain s3 get url stays disposition-free`() {
        s3Service().use { service ->
            val url = service.presignedGetUrl("events/e1/photo.jpg", Duration.ofMinutes(15))
            assertFalse(url.contains("response-content-disposition"))
        }
    }

    // Offline: presigning is pure request signing — no network on build or presign.
    private fun s3Service() = S3StorageService(
        StorageProperties(
            backend = StorageProperties.Backend.S3,
            endpoint = "https://test-account.r2.cloudflarestorage.com",
            accessKey = "test-access",
            secretKey = "test-secret",
            pathStyleAccess = true,
        ),
    )
}
