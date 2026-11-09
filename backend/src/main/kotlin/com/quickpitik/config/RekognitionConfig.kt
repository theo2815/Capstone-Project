package com.quickpitik.config

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.context.annotation.Primary
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.sts.StsClient

// Builds the AWS Rekognition client only when app.ai.provider=rekognition, so
// the ai-api provider never constructs an AWS client. Credentials use the static
// keys from config when present (application-local.yml, gitignored), else the AWS
// default chain (env vars / ~/.aws for dev, instance/task role in prod) — same
// pattern as S3StorageService. Region is explicit from config. Spring auto-closes
// the client (SdkAutoCloseable) on shutdown.
@Configuration
@ConditionalOnProperty(prefix = "app.ai", name = ["provider"], havingValue = "rekognition")
class RekognitionConfig {
    // @Primary: RekognitionAiClient and others inject RekognitionClient by type, and a second
    // bean (liveness, ap-southeast-5) would otherwise fail the context at boot (auto-approval Plan 1, correction 11).
    @Bean
    @Primary
    fun rekognitionClient(props: RekognitionProperties, aiApiProperties: AiApiProperties): RekognitionClient {
        // Batch indexing is hardwired to the concrete ai-api client (mega jobs +
        // webhook/poll ingest), so rekognition + INDEXING_MODE=batch would
        // split-brain: the drain POSTs to ai-api while search reads Rekognition.
        // Refuse to boot rather than run half-broken.
        check(aiApiProperties.indexingMode != IndexingMode.BATCH) {
            "app.ai.provider=rekognition is incompatible with INDEXING_MODE=batch — " +
                "batch drain/ingest is ai-api-only. Use INDEXING_MODE=per_photo."
        }
        return rekognition(props, props.region)
    }

    // AA-I2: Face Liveness sessions (create + read results) in the liveness region.
    @Bean("livenessRekognitionClient")
    fun livenessRekognitionClient(props: RekognitionProperties, identity: IdentityProperties): RekognitionClient =
        rekognition(props, identity.livenessRegion)

    // AA-0.2: the regional endpoint, never the global one; global STS tokens are not
    // valid in an opt-in region such as ap-southeast-5 by default.
    @Bean
    fun livenessStsClient(props: RekognitionProperties, identity: IdentityProperties): StsClient =
        StsClient.builder().region(Region.of(identity.livenessRegion))
            .overrideConfiguration { it.apiCallTimeout(props.operationTimeout) }
            .credentialsProvider(credentials(props))
            .build()

    private fun rekognition(props: RekognitionProperties, region: String): RekognitionClient =
        RekognitionClient.builder().region(Region.of(region))
            .overrideConfiguration { it.apiCallTimeout(props.operationTimeout) }
            .credentialsProvider(credentials(props))
            .build()

    private fun credentials(props: RekognitionProperties): AwsCredentialsProvider =
        if (!props.accessKey.isNullOrBlank() && !props.secretKey.isNullOrBlank()) {
            StaticCredentialsProvider.create(AwsBasicCredentials.create(props.accessKey, props.secretKey))
        } else {
            DefaultCredentialsProvider.create()
        }
}
