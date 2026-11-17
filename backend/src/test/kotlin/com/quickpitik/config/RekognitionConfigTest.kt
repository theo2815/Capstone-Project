package com.quickpitik.config

import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.ai.RekognitionIdentityClient
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.springframework.boot.context.properties.EnableConfigurationProperties
import org.springframework.boot.test.context.runner.ApplicationContextRunner
import org.springframework.context.annotation.Configuration
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.sts.StsClient
import kotlin.test.assertEquals
import kotlin.test.assertNull

class RekognitionConfigTest {
    @Configuration
    @EnableConfigurationProperties(RekognitionProperties::class, AiApiProperties::class, IdentityProperties::class)
    class Props

    private val runner = ApplicationContextRunner()
        .withUserConfiguration(Props::class.java, RekognitionConfig::class.java, RekognitionAiClient::class.java, RekognitionIdentityClient::class.java)
        .withBean(MeterRegistry::class.java, { SimpleMeterRegistry() })
        .withPropertyValues(
            "app.ai.provider=rekognition",
            "app.ai.rekognition.access-key=AKIAFAKEFAKEFAKE",
            "app.ai.rekognition.secret-key=fake-secret",
        )

    @Test
    fun `rekognition provider starts with both rekognition clients`() {
        runner.run { context ->
            assertNull(context.startupFailure)
            val primary = context.getBean(RekognitionClient::class.java)
            val liveness = context.getBean("livenessRekognitionClient", RekognitionClient::class.java)
            assertEquals(Region.AP_SOUTHEAST_1, primary.serviceClientConfiguration().region())
            assertEquals(Region.of("ap-southeast-5"), liveness.serviceClientConfiguration().region())
            // STS must use the regional endpoint: global tokens are not valid in the opt-in region (AA-0.2).
            assertEquals(Region.of("ap-southeast-5"), context.getBean(StsClient::class.java).serviceClientConfiguration().region())
            // Both consumers wired: the face/bib client by type (the @Primary bean), the identity client by name.
            context.getBean(RekognitionAiClient::class.java)
            context.getBean(RekognitionIdentityClient::class.java)
        }
    }
}
