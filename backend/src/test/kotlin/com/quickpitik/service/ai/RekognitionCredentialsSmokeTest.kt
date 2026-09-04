package com.quickpitik.service.ai

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.yaml.snakeyaml.Yaml
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.core.exception.SdkClientException
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.CreateCollectionRequest
import software.amazon.awssdk.services.rekognition.model.DeleteCollectionRequest
import java.io.File

// Live smoke test — proves the AWS Rekognition credentials actually work end to
// end (valid keys + region reachable + create/delete-collection allowed) by
// creating then deleting a throwaway collection. Free (a metadata op, not billed
// per image), non-destructive, in your own AWS account.
//
// Credentials come from app.ai.rekognition.{access-key,secret-key} in
// application-local.yml when present, else the AWS default chain (env vars /
// ~/.aws/credentials) — the same order the app's RekognitionConfig uses, so this
// tests exactly what the running app will do. Skips itself if no creds are found.
//
// Tagged "integration" so the default `test` task ignores it. Run just this one:
//   ./gradlew -p backend integrationTest --tests "*RekognitionCredentialsSmokeTest"
@Tag("integration")
class RekognitionCredentialsSmokeTest {

    @Test
    fun `rekognition credentials can create and delete a collection`() {
        val cfg = readLocalRekognitionConfig()
        val region = cfg?.region ?: "ap-southeast-1"

        val builder = RekognitionClient.builder().region(Region.of(region))
        if (cfg?.accessKey != null && cfg.secretKey != null) {
            builder.credentialsProvider(
                StaticCredentialsProvider.create(AwsBasicCredentials.create(cfg.accessKey, cfg.secretKey)),
            )
            println("Using credentials from application-local.yml (region=$region)")
        } else {
            builder.credentialsProvider(DefaultCredentialsProvider.create())
            println("No keys in application-local.yml; trying the AWS default chain (env / ~/.aws), region=$region")
        }

        val collectionId = "qp-smoke-test-${System.nanoTime()}"
        try {
            builder.build().use { rek ->
                rek.createCollection(CreateCollectionRequest.builder().collectionId(collectionId).build())
                rek.deleteCollection(DeleteCollectionRequest.builder().collectionId(collectionId).build())
            }
        } catch (ex: SdkClientException) {
            // Client-side: no credentials could be resolved, or no network — can't
            // run the check, so skip (not fail) with guidance.
            assumeTrue(false) {
                "No AWS credentials found (or no network). Put access-key/secret-key under " +
                    "app.ai.rekognition in application-local.yml, then re-run. Detail: ${ex.message}"
            }
        }
        // Any AWS *service* error (bad key → UnrecognizedClientException, missing
        // permission → AccessDeniedException) is NOT caught above and FAILS the
        // test with the AWS message — exactly the signal we want to surface.
        println("✅ Rekognition OK — created + deleted $collectionId in $region. Credentials + permissions work.")
    }

    private data class RekCfg(val region: String?, val accessKey: String?, val secretKey: String?)

    @Suppress("UNCHECKED_CAST")
    private fun readLocalRekognitionConfig(): RekCfg? {
        val file = File("src/main/resources/application-local.yml")
        if (!file.exists()) return null
        val root = file.inputStream().use { Yaml().load<Map<String, Any?>>(it) } ?: return null
        val app = root["app"] as? Map<String, Any?> ?: return null
        val rek = (app["ai"] as? Map<String, Any?>)?.get("rekognition") as? Map<String, Any?> ?: return null
        return RekCfg(
            region = rek["region"] as? String,
            accessKey = (rek["access-key"] as? String)?.takeIf { it.isNotBlank() },
            secretKey = (rek["secret-key"] as? String)?.takeIf { it.isNotBlank() },
        )
    }
}
