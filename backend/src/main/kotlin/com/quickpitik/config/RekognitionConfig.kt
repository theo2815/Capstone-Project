package com.quickpitik.config

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.rekognition.RekognitionClient

// Builds the AWS Rekognition client only when app.ai.provider=rekognition, so
// the ai-api provider never constructs an AWS client. Credentials use the static
// keys from config when present (application-local.yml, gitignored), else the AWS
// default chain (env vars / ~/.aws for dev, instance/task role in prod) — same
// pattern as S3StorageService. Region is explicit from config. Spring auto-closes
// the client (SdkAutoCloseable) on shutdown.
@Configuration
@ConditionalOnProperty(prefix = "app.ai", name = ["provider"], havingValue = "rekognition")
class RekognitionConfig {
    @Bean
    fun rekognitionClient(props: RekognitionProperties, aiApiProperties: AiApiProperties): RekognitionClient {
        // Batch indexing is hardwired to the concrete ai-api client (mega jobs +
        // webhook/poll ingest), so rekognition + INDEXING_MODE=batch would
        // split-brain: the drain POSTs to ai-api while search reads Rekognition.
        // Refuse to boot rather than run half-broken.
        check(aiApiProperties.indexingMode != IndexingMode.BATCH) {
            "app.ai.provider=rekognition is incompatible with INDEXING_MODE=batch — " +
                "batch drain/ingest is ai-api-only. Use INDEXING_MODE=per_photo."
        }
        val builder = RekognitionClient.builder().region(Region.of(props.region))
        if (!props.accessKey.isNullOrBlank() && !props.secretKey.isNullOrBlank()) {
            builder.credentialsProvider(
                StaticCredentialsProvider.create(AwsBasicCredentials.create(props.accessKey, props.secretKey)),
            )
        } else {
            builder.credentialsProvider(DefaultCredentialsProvider.create())
        }
        return builder.build()
    }
}
