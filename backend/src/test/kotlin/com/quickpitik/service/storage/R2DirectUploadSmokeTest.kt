package com.quickpitik.service.storage

import com.fasterxml.jackson.databind.ObjectMapper
import com.sun.net.httpserver.HttpServer
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.S3Configuration
import software.amazon.awssdk.services.s3.model.CopyObjectRequest
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.HeadObjectRequest
import software.amazon.awssdk.services.s3.model.MetadataDirective
import software.amazon.awssdk.services.s3.model.S3Exception
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.presigner.S3Presigner
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.InetSocketAddress
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.security.MessageDigest
import java.time.Duration
import java.util.UUID
import java.util.Base64
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import javax.imageio.ImageIO

/** Opt-in live R2 capability probe. Set all R2_SMOKE_* variables for an isolated, private test bucket. */
@Tag("integration")
class R2DirectUploadSmokeTest {
    @Test
    fun `installed SDK signs browser controlled length and generates a near limit JPEG`() {
        val large = jpeg(4000, 3200)
        assertTrue(large.size in 7 * 1024 * 1024..8 * 1024 * 1024) {
            "Generated JPEG size=${large.size}"
        }
        S3Presigner.builder().endpointOverride(URI.create("https://example.r2.cloudflarestorage.com"))
            .region(Region.of("auto"))
            .credentialsProvider(StaticCredentialsProvider.create(AwsBasicCredentials.create("test", "test")))
            .serviceConfiguration(S3Configuration.builder().pathStyleAccessEnabled(true).build())
            .build().use { presigner ->
                val uri = presigner.presignPutObject(PutObjectPresignRequest.builder()
                    .signatureDuration(Duration.ofMinutes(5))
                    .putObjectRequest(PutObjectRequest.builder().bucket("test-bucket").key("test.jpg")
                        .contentType("image/jpeg").contentLength(large.size.toLong()).build())
                    .build()).url().toURI()
                assertEquals("/test-bucket/test.jpg", uri.path)
                val signed = uri.rawQuery
                    .split('&').first { it.startsWith("X-Amz-SignedHeaders=") }.substringAfter('=')
                assertTrue(signed.contains("content-type") && signed.contains("content-length")) {
                    "SDK signed headers: $signed"
                }
            }
    }

    @Test
    @Suppress("DEPRECATION") // CopyObjectRequest's installed SDK builder still exposes these R2-compatible fields.
    fun `signed size type and create only copy hold on the test bucket`() {
        val bucket = System.getenv("R2_SMOKE_BUCKET")
        assumeTrue(!bucket.isNullOrBlank(), "Set R2_SMOKE_BUCKET to an isolated test bucket")
        require(!bucket!!.contains("prod", ignoreCase = true)) { "Production buckets are forbidden for this probe" }
        val endpoint = URI.create(requireEnv("R2_SMOKE_ENDPOINT"))
        require(endpoint.scheme == "https" && endpoint.host.endsWith(".r2.cloudflarestorage.com"))
        val credentials = StaticCredentialsProvider.create(AwsBasicCredentials.create(
            requireEnv("R2_SMOKE_ACCESS_KEY"), requireEnv("R2_SMOKE_SECRET_KEY"),
        ))
        val config = S3Configuration.builder().pathStyleAccessEnabled(true).build()
        val client = S3Client.builder().endpointOverride(endpoint).region(Region.of("auto"))
            .credentialsProvider(credentials).serviceConfiguration(config).build()
        val presigner = S3Presigner.builder().endpointOverride(endpoint).region(Region.of("auto"))
            .credentialsProvider(credentials).serviceConfiguration(config).build()
        val http = HttpClient.newHttpClient()
        val prefix = "smoke-test/direct-upload/${UUID.randomUUID()}"
        val created = mutableListOf<String>()
        val origin = requireEnv("R2_SMOKE_ORIGIN")
        val corsOrigins = mutableListOf<String?>()
        fun key(name: String) = "$prefix/$name".also { created.add(it) }
        try {
            val small = jpeg(64, 48)
            val large = jpeg(4000, 3200)
            assertTrue(large.size in 7 * 1024 * 1024..8 * 1024 * 1024) {
                "Generated JPEG must exercise the website's 7-8 MiB boundary; actual=${large.size}"
            }
            listOf("small" to small, "large" to large).forEach { (name, bytes) ->
                val source = key("$name-stage.jpg")
                val url = presigner.presignPutObject(PutObjectPresignRequest.builder()
                    .signatureDuration(Duration.ofMinutes(5))
                    .putObjectRequest(PutObjectRequest.builder().bucket(bucket).key(source)
                        .contentType("image/jpeg").contentLength(bytes.size.toLong()).build())
                    .build())
                val signed = URI.create(url.url().toString()).rawQuery
                    .split('&').first { it.startsWith("X-Amz-SignedHeaders=") }.substringAfter('=')
                assertTrue(signed.contains("content-type") && signed.contains("content-length")) {
                    "SDK did not sign both type and length: $signed"
                }
                val uri = url.url().toURI() // Never print: this URL is a write capability.
                fun put(body: ByteArray, type: String) = http.send(
                    HttpRequest.newBuilder(uri).header("Content-Type", type)
                        .PUT(HttpRequest.BodyPublishers.ofByteArray(body)).build(),
                    HttpResponse.BodyHandlers.ofString(),
                )
                val wrongType = put(bytes, "image/png")
                assertEquals(403, wrongType.statusCode(), "Wrong type was accepted")
                assertTrue(wrongType.body().contains("SignatureDoesNotMatch"), "Unexpected R2 wrong-type error")
                assertMissing(client, bucket, source)
                val shortBody = put(bytes.copyOf(bytes.size - 1), "image/jpeg")
                assertEquals(403, shortBody.statusCode(), "Short body was accepted")
                assertTrue(shortBody.body().contains("SignatureDoesNotMatch"), "Unexpected R2 short-body error")
                assertMissing(client, bucket, source)
                val longBody = put(bytes + 0, "image/jpeg")
                assertEquals(403, longBody.statusCode(), "Long body was accepted")
                assertTrue(longBody.body().contains("SignatureDoesNotMatch"), "Unexpected R2 long-body error")
                assertMissing(client, bucket, source)
                assertEquals(200, put(bytes, "image/jpeg").statusCode())
                assertArrayEquals(sha(bytes), sha(client.getObjectAsBytes(
                    GetObjectRequest.builder().bucket(bucket).key(source).build()).asByteArray()))
                val anonymousRead = http.send(HttpRequest.newBuilder(
                    URI.create("$endpoint/$bucket/$source")).GET().build(), HttpResponse.BodyHandlers.ofString())
                assertTrue(anonymousRead.statusCode() in 400..499) {
                    "Unexpected anonymous object-read status ${anonymousRead.statusCode()}"
                }

                val final = key("$name-final.jpg")
                val etag = client.headObject(HeadObjectRequest.builder().bucket(bucket).key(source).build()).eTag()
                val preflight = http.send(HttpRequest.newBuilder(uri)
                    .header("Origin", origin).header("Access-Control-Request-Method", "PUT")
                    .header("Access-Control-Request-Headers", "content-type")
                    .method("OPTIONS", HttpRequest.BodyPublishers.noBody()).build(),
                    HttpResponse.BodyHandlers.discarding())
                corsOrigins.add(preflight.headers().firstValue("Access-Control-Allow-Origin").orElse(null))
                fun copy(from: String, to: String, match: String) = client.copyObject(
                    CopyObjectRequest.builder().bucket(bucket).key(to).copySource("$bucket/$from")
                        .copySourceIfMatch(match).metadataDirective(MetadataDirective.REPLACE)
                        .contentType("image/jpeg").cacheControl("private, no-store")
                        .overrideConfiguration { it.putHeader("cf-copy-destination-if-none-match", "*") }
                        .build(),
                )
                copy(source, final, etag) // Simulate a lost response: recover by reading final.
                assertArrayEquals(sha(bytes), sha(client.getObjectAsBytes(
                    GetObjectRequest.builder().bucket(bucket).key(final).build()).asByteArray()))
                if (name == "small") {
                    val racing = key("racing-final.jpg")
                    val pool = Executors.newFixedThreadPool(2)
                    try {
                        val ready = CountDownLatch(2)
                        val start = CountDownLatch(1)
                        val results = (1..2).map { pool.submit<Int> {
                            ready.countDown()
                            check(start.await(10, TimeUnit.SECONDS))
                            runCatching { copy(source, racing, etag); 200 }
                                .getOrElse { (it as S3Exception).statusCode() }
                        } }
                        assertTrue(ready.await(10, TimeUnit.SECONDS))
                        start.countDown()
                        val statuses = results.map { it.get() }
                        assertEquals(listOf(200, 412), statuses.sorted())
                    } finally { pool.shutdownNow() }
                    assertArrayEquals(sha(bytes), sha(client.getObjectAsBytes(
                        GetObjectRequest.builder().bucket(bucket).key(racing).build()).asByteArray()))
                }
                val changed = bytes.copyOf().also { it[it.lastIndex] = (it.last().toInt() xor 1).toByte() }
                assertEquals(200, put(changed, "image/jpeg").statusCode())
                val staleSourceTarget = key("$name-stale-source.jpg")
                assertEquals(412, runCatching { copy(source, staleSourceTarget, etag) }
                    .exceptionOrNull().let { (it as? S3Exception)?.statusCode() })
                assertMissing(client, bucket, staleSourceTarget)
                assertEquals(412, runCatching { copy(source, final, etag) }
                    .exceptionOrNull().let { (it as? S3Exception)?.statusCode() })
                assertArrayEquals(sha(bytes), sha(client.getObjectAsBytes(
                    GetObjectRequest.builder().bucket(bucket).key(final).build()).asByteArray()))
            }
            val native = key("native-stage.jpg")
            val nativeUrl = presigner.presignPutObject(PutObjectPresignRequest.builder()
                .signatureDuration(Duration.ofMinutes(5))
                .putObjectRequest(PutObjectRequest.builder().bucket(bucket).key(native)
                    .contentType("image/jpeg").contentLength(small.size.toLong()).build())
                .build()).url().toString()
            val file = Files.createTempFile("r2-native-smoke-", ".jpg").toFile()
            try {
                file.writeBytes(small)
                // Exact request construction used by mobile/PhotoUploadWorker.kt.
                val put = Request.Builder().url(nativeUrl)
                    .put(file.asRequestBody("image/jpeg".toMediaTypeOrNull())).build()
                OkHttpClient().newCall(put).execute().use { assertEquals(200, it.code) }
                assertArrayEquals(sha(small), sha(client.getObjectAsBytes(
                    GetObjectRequest.builder().bucket(bucket).key(native).build()).asByteArray()))
            } finally { file.delete() }
            assertTrue(corsOrigins.all { it == origin }) { "R2 preflight did not allow the approved origin" }
            val browser = key("browser-stage.jpg")
            val browserUrl = presigner.presignPutObject(PutObjectPresignRequest.builder()
                .signatureDuration(Duration.ofMinutes(5))
                .putObjectRequest(PutObjectRequest.builder().bucket(bucket).key(browser)
                    .contentType("image/jpeg").contentLength(small.size.toLong()).build())
                .build()).url().toString()
            browserPut(browserUrl, small, origin)
            assertArrayEquals(sha(small), sha(client.getObjectAsBytes(
                GetObjectRequest.builder().bucket(bucket).key(browser).build()).asByteArray()))
        } finally {
            created.distinct().forEach { objectKey ->
                client.deleteObject(DeleteObjectRequest.builder().bucket(bucket).key(objectKey).build())
            }
            presigner.close()
            client.close()
        }
    }

    private fun requireEnv(name: String) = System.getenv(name)?.takeIf { it.isNotBlank() }
        ?: error("Set $name for the isolated R2 smoke test")

    private fun sha(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)

    private fun browserPut(url: String, bytes: ByteArray, origin: String) {
        require(origin == "http://localhost:3000") { "This browser probe serves only the approved local origin" }
        val completed = CountDownLatch(1)
        val status = AtomicReference<String>()
        val server = HttpServer.create(InetSocketAddress("localhost", 3000), 0)
        fun respond(exchange: com.sun.net.httpserver.HttpExchange, contentType: String, body: ByteArray) {
            exchange.responseHeaders.add("Content-Type", contentType)
            exchange.sendResponseHeaders(200, body.size.toLong())
            exchange.responseBody.use { it.write(body) }
        }
        server.createContext("/grant") { exchange ->
            respond(exchange, "application/json", ObjectMapper().writeValueAsBytes(mapOf(
                "url" to url, "bytes" to Base64.getEncoder().encodeToString(bytes),
            )))
        }
        server.createContext("/done") { exchange ->
            status.set(exchange.requestURI.rawQuery?.substringAfter("status="))
            completed.countDown()
            respond(exchange, "text/plain", "ok".toByteArray())
        }
        server.createContext("/") { exchange ->
            respond(exchange, "text/html", """
                <!doctype html><body><script>
                fetch('/grant').then(r => r.json()).then(({url, bytes}) => {
                  const data = Uint8Array.from(atob(bytes), c => c.charCodeAt(0));
                  const file = new File([data], 'smoke.jpg', {type: 'image/jpeg'});
                  const xhr = new XMLHttpRequest();
                  xhr.open('PUT', url);
                  xhr.setRequestHeader('Content-Type', 'image/jpeg');
                  xhr.onload = () => fetch('/done?status=' + xhr.status + ':' + !!xhr.getResponseHeader('ETag'));
                  xhr.onerror = () => fetch('/done?status=network');
                  xhr.send(file);
                });
                </script></body>
            """.trimIndent().toByteArray())
        }
        val profile = Files.createTempDirectory("r2-browser-smoke-").toFile()
        val chrome = File(System.getenv("PROGRAMFILES"), "Google/Chrome/Application/chrome.exe")
        var process: Process? = null
        try {
            assertTrue(chrome.isFile, "Chrome is required for the browser XHR smoke")
            server.start()
            process = ProcessBuilder(chrome.path, "--headless=new", "--disable-gpu", "--no-first-run",
                "--no-default-browser-check", "--user-data-dir=${profile.path}", origin)
                .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                .redirectError(ProcessBuilder.Redirect.DISCARD).start()
            assertTrue(completed.await(30, TimeUnit.SECONDS), "Browser XHR did not finish")
            assertEquals("200:true", status.get(), "Browser XHR or exposed ETag failed")
        } finally {
            process?.destroyForcibly()
            process?.waitFor(5, TimeUnit.SECONDS)
            server.stop(0)
            check(profile.canonicalFile.toPath().startsWith(File(System.getProperty("java.io.tmpdir")).canonicalFile.toPath()))
            profile.deleteRecursively()
        }
    }

    private fun assertMissing(client: S3Client, bucket: String, key: String) {
        assertEquals(404, runCatching { client.headObject(
            HeadObjectRequest.builder().bucket(bucket).key(key).build()) }
            .exceptionOrNull().let { (it as? S3Exception)?.statusCode() })
    }

    private fun jpeg(width: Int, height: Int): ByteArray {
        val image = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val random = java.util.Random(123)
        for (y in 0 until height) for (x in 0 until width) image.setRGB(x, y, random.nextInt(0x1000000))
        return ByteArrayOutputStream().also { ImageIO.write(image, "jpg", it) }.toByteArray()
    }
}
