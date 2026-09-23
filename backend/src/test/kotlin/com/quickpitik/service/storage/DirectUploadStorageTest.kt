package com.quickpitik.service.storage

import com.quickpitik.config.StorageProperties
import com.sun.net.httpserver.HttpServer
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.springframework.mock.env.MockEnvironment
import java.net.InetSocketAddress
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.nio.file.FileAlreadyExistsException
import java.nio.file.NoSuchFileException
import java.nio.file.Path
import java.time.Duration
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DirectUploadStorageTest {

    @TempDir
    lateinit var tempDir: Path

    @Test
    fun `strict presign binds content length and type without returning forbidden headers`() {
        s3Service().use { service ->
            val upload = service.presignedUpload(
                "staging/photo-uploads/owner/event/photo/original",
                Duration.ofMinutes(15),
                "image/jpeg",
                7_340_032,
            )

            val query = URLDecoder.decode(upload.url.substringAfter('?'), StandardCharsets.UTF_8)
            assertTrue(query.contains("X-Amz-SignedHeaders=content-length;content-type;host"), query)
            assertTrue(query.contains("X-Amz-Signature="), query)
            assertEquals(mapOf("Content-Type" to "image/jpeg"), upload.headers)
        }
    }

    @Test
    fun `s3 copy sends both source and destination preconditions`() {
        val copyHeaders = AtomicReference<Map<String, String>>()
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            createContext("/") { exchange ->
                if (exchange.requestMethod == "HEAD") {
                    exchange.responseHeaders.add("Content-Length", "123")
                    exchange.responseHeaders.add("Content-Type", "image/jpeg")
                    exchange.responseHeaders.add("Content-Encoding", "identity")
                    exchange.responseHeaders.add("ETag", "\"winner\"")
                    exchange.sendResponseHeaders(200, -1)
                } else {
                    copyHeaders.set(
                        exchange.requestHeaders.entries.associate { (name, values) ->
                            name.lowercase() to values.single()
                        },
                    )
                    val body = """
                        <CopyObjectResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
                          <LastModified>2026-09-23T00:00:00Z</LastModified>
                          <ETag>&quot;winner&quot;</ETag>
                        </CopyObjectResult>
                    """.trimIndent().toByteArray()
                    exchange.responseHeaders.add("Content-Type", "application/xml")
                    exchange.sendResponseHeaders(200, body.size.toLong())
                    exchange.responseBody.use { it.write(body) }
                }
                exchange.close()
            }
            start()
        }

        try {
            s3Service("http://127.0.0.1:${server.address.port}").use { service ->
                val stored = service.copyIfAbsent(
                    "staging/photo-uploads/owner/event/photo/original",
                    "events/event/photos/photo/original.jpg",
                    "\"source-etag\"",
                    "image/jpeg",
                )

                assertEquals(123, stored.sizeBytes)
                assertEquals("\"winner\"", stored.etag)
                assertEquals("identity", stored.contentEncoding)
            }

            val headers = copyHeaders.get()
            assertEquals("\"source-etag\"", headers["x-amz-copy-source-if-match"])
            assertEquals("*", headers["cf-copy-destination-if-none-match"])
            assertEquals("REPLACE", headers["x-amz-metadata-directive"])
            assertEquals("private, no-store", headers["cache-control"])
            val source = URLDecoder.decode(headers.getValue("x-amz-copy-source"), StandardCharsets.UTF_8)
            assertTrue(source.contains("quickpitik-dev/staging/photo-uploads"), source)
        } finally {
            server.stop(0)
        }
    }

    @Test
    fun `local create-only copy never overwrites an existing target`() {
        val service = LocalFsStorageService(
            StorageProperties(localRoot = tempDir.toString()),
            MockEnvironment(),
        )
        val source = "staging/photo-uploads/source"
        val target = "events/event/photos/photo/original.jpg"
        service.put(source, byteArrayOf(1, 2, 3), "image/jpeg")
        service.put(target, byteArrayOf(9), "image/jpeg")

        assertFailsWith<FileAlreadyExistsException> {
            service.copyIfAbsent(source, target, "ignored-locally", "image/jpeg")
        }
        assertContentEquals(byteArrayOf(9), service.getBytes(target))

        val fresh = "events/event/photos/fresh/original.jpg"
        service.copyIfAbsent(source, fresh, "ignored-locally", "image/jpeg")
        assertContentEquals(byteArrayOf(1, 2, 3), service.getBytes(fresh))
    }

    @Test
    fun `failed local copy leaves no partial destination`() {
        val service = LocalFsStorageService(
            StorageProperties(localRoot = tempDir.toString()),
            MockEnvironment(),
        )
        val target = "events/event/photos/photo/original.jpg"

        assertFailsWith<NoSuchFileException> {
            service.copyIfAbsent("staging/missing", target, "ignored-locally", "image/jpeg")
        }
        assertFalse(service.exists(target))
    }

    @Test
    fun `immutable direct capability requires an enabled R2 endpoint`() {
        s3Service().use { assertFalse(it.supportsImmutableDirectUpload) }
        s3Service(
            directUpload = StorageProperties.DirectUpload(enabled = true),
        ).use { assertTrue(it.supportsImmutableDirectUpload) }
    }

    private fun s3Service(
        endpoint: String = "https://test-account.r2.cloudflarestorage.com",
        directUpload: StorageProperties.DirectUpload = StorageProperties.DirectUpload(),
    ) = S3StorageService(
        StorageProperties(
            backend = StorageProperties.Backend.S3,
            endpoint = endpoint,
            accessKey = "test-access",
            secretKey = "test-secret",
            pathStyleAccess = true,
            directUpload = directUpload,
        ),
    )
}
