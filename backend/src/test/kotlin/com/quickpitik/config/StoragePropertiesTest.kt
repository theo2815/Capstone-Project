package com.quickpitik.config

import org.junit.jupiter.api.Test
import java.time.Duration
import kotlin.test.assertFailsWith

class StoragePropertiesTest {

    @Test
    fun `website direct upload cannot be enabled without the global gate`() {
        assertFailsWith<IllegalArgumentException> {
            StorageProperties(
                directUpload = StorageProperties.DirectUpload(websiteEnabled = true),
            )
        }
    }

    @Test
    fun `direct upload requires an R2 endpoint`() {
        assertFailsWith<IllegalArgumentException> {
            StorageProperties(
                backend = StorageProperties.Backend.S3,
                endpoint = "https://s3.ap-southeast-1.amazonaws.com",
                directUpload = StorageProperties.DirectUpload(enabled = true),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            StorageProperties(
                backend = StorageProperties.Backend.S3,
                endpoint = "http://test-account.r2.cloudflarestorage.com",
                directUpload = StorageProperties.DirectUpload(enabled = true),
            )
        }
    }

    @Test
    fun `intent and lease durations preserve recovery budgets`() {
        assertFailsWith<IllegalArgumentException> {
            StorageProperties(
                backend = StorageProperties.Backend.S3,
                endpoint = "https://test-account.r2.cloudflarestorage.com",
                directUpload = StorageProperties.DirectUpload(
                    enabled = true,
                    putTtl = Duration.ofMinutes(15),
                    intentTtl = Duration.ofMinutes(15),
                ),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            StorageProperties(
                backend = StorageProperties.Backend.S3,
                endpoint = "https://test-account.r2.cloudflarestorage.com",
                directUpload = StorageProperties.DirectUpload(
                    enabled = true,
                    verificationLease = Duration.ofSeconds(120),
                ),
            )
        }
    }
}
