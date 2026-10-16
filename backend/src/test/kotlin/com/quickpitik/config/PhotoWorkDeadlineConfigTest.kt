package com.quickpitik.config

import com.quickpitik.service.storage.S3StorageService
import org.junit.jupiter.api.Test
import org.springframework.boot.context.properties.bind.Bindable
import org.springframework.boot.context.properties.bind.BindException
import org.springframework.boot.context.properties.bind.Binder
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource
import org.springframework.test.util.ReflectionTestUtils
import software.amazon.awssdk.services.s3.S3Client
import java.time.Duration
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PhotoWorkDeadlineConfigTest {
    private fun <T : Any> bind(prefix: String, type: Class<T>, values: Map<String, String>): T =
        Binder(MapConfigurationPropertySource(values)).bind(prefix, Bindable.of(type)).get()

    @Test fun `storage SDK honors the configured total operation deadline`() {
        val props = bind("app.storage", StorageProperties::class.java, mapOf(
            "app.storage.operation-timeout" to "PT37S", "app.storage.access-key" to "synthetic",
            "app.storage.secret-key" to "synthetic", "app.storage.region" to "ap-southeast-1",
        ))
        S3StorageService(props).use { storage ->
            val client = ReflectionTestUtils.getField(storage, "client") as S3Client
            assertEquals(Duration.ofSeconds(37), client.serviceClientConfiguration().overrideConfiguration().apiCallTimeout().orElse(null))
        }
    }

    @Test fun `Rekognition SDK honors the configured total operation deadline`() {
        val props = bind("app.ai.rekognition", RekognitionProperties::class.java, mapOf(
            "app.ai.rekognition.operation-timeout" to "PT41S", "app.ai.rekognition.access-key" to "synthetic",
            "app.ai.rekognition.secret-key" to "synthetic", "app.ai.rekognition.region" to "ap-southeast-1",
        ))
        RekognitionConfig().rekognitionClient(props, AiApiProperties()).use { client ->
            assertEquals(Duration.ofSeconds(41), client.serviceClientConfiguration().overrideConfiguration().apiCallTimeout().orElse(null))
        }
    }

    @Test fun `nonpositive deadlines are rejected during configuration binding`() {
        for (value in listOf("PT0S", "-PT1S")) {
            assertFailsWith<BindException> {
                bind("app.storage", StorageProperties::class.java,
                    mapOf("app.storage.region" to "ap-southeast-1", "app.storage.operation-timeout" to value))
            }
            assertFailsWith<BindException> {
                bind("app.ai.rekognition", RekognitionProperties::class.java,
                    mapOf("app.ai.rekognition.region" to "ap-southeast-1", "app.ai.rekognition.operation-timeout" to value))
            }
        }
    }
}
