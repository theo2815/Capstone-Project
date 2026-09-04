package com.quickpitik.service.ai

import com.quickpitik.config.RekognitionProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.service.storage.S3StorageService
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.yaml.snakeyaml.Yaml
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.rekognition.RekognitionClient
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.UUID

// Live end-to-end smoke test of the actual migration code paths against REAL AWS
// Rekognition + REAL Cloudflare R2, using the real production classes
// (S3StorageService + RekognitionAiClient) and config read from
// application-local.yml. It uses real images already in .storage (dev disk) and
// cleans up after itself (deletes the Rekognition collection + the R2 test
// object). Everything self-skips when its config/images are absent.
//
// This covers everything the migration touches — R2 read/write, IndexFaces,
// SearchFacesByImage (a real cross-image face match), DetectText — but NOT the
// unchanged HTTP/auth/upload/event plumbing (that needs the full app + Postgres).
//
// Tagged "integration" so the default `test` task skips it. Run it with:
//   ./gradlew -p backend integrationTest --tests "*RekognitionR2E2eSmokeTest"
@Tag("integration")
class RekognitionR2E2eSmokeTest {

    @Test
    fun `R2 storage put get presign delete round-trips`() {
        val storage = storageServiceOrNull()
        assumeTrue(storage != null) {
            "No R2 storage config in application-local.yml (need app.storage.backend=S3 + endpoint) — skipping"
        }
        storage!!.use { s ->
            val key = "smoke-test/${UUID.randomUUID()}.txt"
            val bytes = "quickpitik-r2-smoke".toByteArray()
            s.put(key, bytes, "text/plain")
            val fetched = s.getBytes(key)
            assertTrue(fetched.contentEquals(bytes)) { "R2 getBytes did not match what was put" }
            val url = s.presignedGetUrl(key, Duration.ofMinutes(5))
            assertTrue(url.startsWith("https://")) { "presigned URL is not https" }
            assertTrue(s.exists(key)) { "R2 object should exist after put" }
            // Signed response-content-disposition must be ACCEPTED live — an
            // unsigned appended param 403s SignatureDoesNotMatch, which is the
            // failure mode presignedDownloadUrl exists to avoid.
            val dlUrl = s.presignedDownloadUrl(key, Duration.ofMinutes(5), "smoke-download.txt")
            val resp = HttpClient.newHttpClient().send(
                HttpRequest.newBuilder(URI.create(dlUrl)).GET().build(),
                HttpResponse.BodyHandlers.ofByteArray(),
            )
            assertTrue(resp.statusCode() == 200) { "signed-disposition GET returned ${resp.statusCode()}" }
            val cd = resp.headers().firstValue("Content-Disposition").orElse("")
            assertTrue(cd.contains("attachment")) { "Content-Disposition missing or inline: '$cd'" }
            s.delete(key)
            println("✅ R2 OK — put + get + presigned GET + signed-disposition download (header: $cd) + delete round-tripped.")
        }
    }

    @Test
    fun `Rekognition enrolls one selfie and face-search matches a different selfie of the same person`() {
        val (rek, props) = rekOrNull() ?: run {
            assumeTrue(false) { "No app.ai.rekognition config / AWS credentials — skipping" }
            return
        }
        val pair = findSelfiePair()
        assumeTrue(pair != null) {
            "Need >=2 selfies of one person under .storage/selfies/<id>/ — skipping (none found)"
        }
        val (enrollImg, searchImg) = pair!!
        val client = RekognitionAiClient(rek, props)
        val eventId = UUID.randomUUID()
        val photoId = UUID.randomUUID()
        try {
            val enroll = client.facesEnroll(
                file = enrollImg.readBytes(),
                contentType = "image/jpeg",
                filename = "$photoId.jpg",
                personName = photoId.toString(),
                personId = null,
                eventId = eventId,
            )
            assertTrue(enroll.faces_enrolled >= 1) { "IndexFaces found no face in ${enrollImg.name}" }
            println("Enrolled ${enroll.faces_enrolled} face(s) from ${enrollImg.name} as ${enroll.person_id}")

            // Retry a few times — a freshly indexed face can lag briefly before search sees it.
            var matched = false
            var bestSim = 0.0
            repeat(5) { attempt ->
                if (!matched) {
                    val result = client.facesSearch(
                        file = searchImg.readBytes(),
                        contentType = "image/jpeg",
                        filename = "selfie.jpg",
                        eventId = eventId,
                        threshold = props.faceMatchThreshold,
                        topK = 10,
                    )
                    val hit = result.matches.firstOrNull { it.person_id == enroll.person_id }
                    if (hit != null) {
                        matched = true
                        bestSim = hit.similarity
                    } else if (attempt < 4) {
                        Thread.sleep(1000)
                    }
                }
            }
            assertTrue(matched) {
                "Search with ${searchImg.name} did NOT match the enrolled ${enrollImg.name} " +
                    "at threshold ${props.faceMatchThreshold} (0-1). If these are different people that is correct; " +
                    "if the same person, the two selfies may be too dissimilar."
            }
            println(
                "✅ Rekognition face match OK — ${searchImg.name} matched the enrolled ${enrollImg.name} " +
                    "at similarity ${"%.3f".format(bestSim)} (>= ${props.faceMatchThreshold}).",
            )
        } finally {
            client.deleteFacesByEvent(eventId) // drop the throwaway collection
        }
    }

    @Test
    fun `Rekognition DetectText runs on a race photo`() {
        val (rek, props) = rekOrNull() ?: run {
            assumeTrue(false) { "No app.ai.rekognition config / AWS credentials — skipping" }
            return
        }
        val photo = findEventPhoto()
        assumeTrue(photo != null) { "No .storage/events/*/photos/*/original.jpg found — skipping" }
        val client = RekognitionAiClient(rek, props)
        val result = client.bibsRecognize(photo!!.readBytes(), "image/jpeg", photo.name, minChars = 1)
        val bibs = result.detections.joinToString(", ") { "${it.bib_number} (${"%.2f".format(it.confidence)})" }
        println("✅ DetectText OK on ${photo.name} — bib-like tokens: ${bibs.ifEmpty { "(none found in this photo)" }}")
        // No assertion on content — proves the DetectText permission + parsing run;
        // whether a given photo contains a legible bib is a separate accuracy question.
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    private fun storageServiceOrNull(): S3StorageService? {
        val s = nested(loadYaml(), "app", "storage") ?: return null
        if ((s["backend"] as? String)?.uppercase() != "S3") return null
        val endpoint = (s["endpoint"] as? String)?.takeIf { it.isNotBlank() } ?: return null
        val accessKey = (s["access-key"] as? String)?.takeIf { it.isNotBlank() } ?: return null
        val secretKey = (s["secret-key"] as? String)?.takeIf { it.isNotBlank() } ?: return null
        return S3StorageService(
            StorageProperties(
                backend = StorageProperties.Backend.S3,
                bucket = (s["bucket"] as? String) ?: "quickpitik-dev",
                region = (s["region"] as? String) ?: "auto",
                endpoint = endpoint,
                accessKey = accessKey,
                secretKey = secretKey,
                pathStyleAccess = (s["path-style-access"] as? Boolean) ?: true,
            ),
        )
    }

    private fun rekOrNull(): Pair<RekognitionClient, RekognitionProperties>? {
        val r = nested(loadYaml(), "app", "ai", "rekognition")
        val region = (r?.get("region") as? String) ?: "ap-southeast-1"
        val accessKey = (r?.get("access-key") as? String)?.takeIf { it.isNotBlank() }
        val secretKey = (r?.get("secret-key") as? String)?.takeIf { it.isNotBlank() }
        val builder = RekognitionClient.builder().region(Region.of(region))
        builder.credentialsProvider(
            if (accessKey != null && secretKey != null) {
                StaticCredentialsProvider.create(AwsBasicCredentials.create(accessKey, secretKey))
            } else {
                DefaultCredentialsProvider.create()
            },
        )
        return builder.build() to RekognitionProperties(region = region)
    }

    private fun findSelfiePair(): Pair<File, File>? =
        File(".storage/selfies").listFiles()?.asSequence()
            ?.filter { it.isDirectory }
            ?.mapNotNull { dir ->
                dir.listFiles { f -> f.isFile && f.extension.equals("jpg", ignoreCase = true) }
                    ?.sortedBy { it.name }?.takeIf { it.size >= 2 }
            }
            ?.firstOrNull()
            ?.let { it[0] to it[1] }

    private fun findEventPhoto(): File? =
        File(".storage/events").takeIf { it.isDirectory }
            ?.walkTopDown()?.firstOrNull { it.isFile && it.name == "original.jpg" }

    @Suppress("UNCHECKED_CAST")
    private fun loadYaml(): Map<String, Any?> {
        val file = File("src/main/resources/application-local.yml")
        if (!file.exists()) return emptyMap()
        return file.inputStream().use { Yaml().load<Map<String, Any?>>(it) } ?: emptyMap()
    }

    @Suppress("UNCHECKED_CAST")
    private fun nested(root: Map<String, Any?>, vararg keys: String): Map<String, Any?>? {
        var cur: Any? = root
        for (k in keys) cur = (cur as? Map<String, Any?>)?.get(k)
        return cur as? Map<String, Any?>
    }
}
