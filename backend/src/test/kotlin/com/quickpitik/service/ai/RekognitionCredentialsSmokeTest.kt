package com.quickpitik.service.ai

import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.yaml.snakeyaml.Yaml
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.core.exception.SdkClientException
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.CreateCollectionRequest
import software.amazon.awssdk.services.rekognition.model.DeleteCollectionRequest
import java.io.File
import kotlin.test.assertTrue

// Live smoke test — proves the AWS Rekognition credentials actually work end to
// end (valid keys + region reachable + create/delete-collection allowed) by
// creating then deleting a throwaway collection. Free (a metadata op, not billed
// per image), non-destructive, in your own AWS account.
//
// Credentials come only from app.ai.rekognition.{access-key,secret-key} in
// application-local.yml. Without them (a worktree, CI) the test skips at once and
// sends nothing to AWS: the machine's own default credentials are never tried.
//
// Tagged "integration" so the default `test` task ignores it. Run just this one:
//   ./gradlew -p backend integrationTest --tests "*RekognitionCredentialsSmokeTest"
@Tag("integration")
class RekognitionCredentialsSmokeTest {

    @Test
    fun `rekognition credentials can create and delete a collection`() {
        val cfg = readLocalRekognitionConfig()
        if (cfg?.accessKey == null || cfg.secretKey == null) {
            assumeTrue(false) {
                "No AWS credentials found. Put access-key/secret-key under " +
                    "app.ai.rekognition in application-local.yml, then re-run."
            }
            return
        }
        val region = cfg.region ?: "ap-southeast-1"

        val builder = RekognitionClient.builder().region(Region.of(region))
            .credentialsProvider(StaticCredentialsProvider.create(AwsBasicCredentials.create(cfg.accessKey, cfg.secretKey)))
        println("Using credentials from application-local.yml (region=$region)")

        val collectionId = "qp-smoke-test-${System.nanoTime()}"
        try {
            builder.build().use { rek ->
                // Asserted on the responses rather than a ListCollections call:
                // the app never lists, so its IAM policy need not allow it.
                val created = rek.createCollection(CreateCollectionRequest.builder().collectionId(collectionId).build())
                rek.deleteCollection(DeleteCollectionRequest.builder().collectionId(collectionId).build())
                assertTrue(created.collectionArn().endsWith("/$collectionId"), created.collectionArn())
            }
        } catch (ex: SdkClientException) {
            // Client-side: no network — can't run the check, so skip (not fail).
            assumeTrue(false) { "AWS could not be reached (no network?). Detail: ${ex.message}" }
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
