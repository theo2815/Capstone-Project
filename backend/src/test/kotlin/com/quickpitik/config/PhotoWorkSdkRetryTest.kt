package com.quickpitik.config

import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.storage.S3StorageService
import com.sun.net.httpserver.HttpServer
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertAll
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.s3.model.S3Exception
import java.net.InetSocketAddress
import java.net.URI
import java.time.Duration
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertIs

class PhotoWorkSdkRetryTest {
    @Test fun `S3 byte PUT cannot hide an earlier failed attempt`() = storageWrite(false)
    @Test fun `S3 stream PUT cannot hide an earlier failed attempt`() = storageWrite(true)
    @Test fun `Rekognition index write cannot hide an earlier failed attempt`() = rekognitionWrite(false)
    @Test fun `Rekognition index write cannot hide an earlier timed out attempt`() = rekognitionWrite(true)

    private fun storageWrite(stream: Boolean) = server { server, requests ->
        val reads = AtomicInteger()
        server.createContext("/") { exchange ->
            exchange.use {
                it.requestBody.readAllBytes()
                val first = (if (it.requestMethod == "GET") reads else requests).incrementAndGet() == 1
                val body = if (first) "<Error><Code>InternalError</Code><Message>uncertain</Message></Error>" else ""
                it.responseHeaders.add("ETag", "synthetic-etag")
                it.sendResponseHeaders(if (first) 503 else 200, if (first) body.length.toLong() else -1)
                if (first) it.responseBody.write(body.toByteArray())
            }
        }
        server.start()
        S3StorageService(StorageProperties(endpoint = "http://127.0.0.1:${server.address.port}",
            bucket = "synthetic", accessKey = "synthetic", secretKey = "synthetic", operationTimeout = Duration.ofSeconds(3))).use { storage ->
            val outcome = runCatching {
                if (stream) storage.put("key", byteArrayOf(1).inputStream(), 1, "image/jpeg")
                else storage.put("key", byteArrayOf(1), "image/jpeg")
            }
            assertAll({ assertIs<S3Exception>(outcome.exceptionOrNull()) }, { assertEquals(1, requests.get()) })
            assertEquals(0, storage.getBytes("key").size)
            assertEquals(2, reads.get(), "A write override must not disable GET retries on the same client")
        }
    }

    private fun rekognitionWrite(timeout: Boolean) = server { server, requests ->
        val reads = AtomicInteger()
        server.createContext("/") { exchange ->
            exchange.use {
                it.requestBody.readAllBytes()
                val target = it.requestHeaders.getFirst("X-Amz-Target")
                val indexing = target.endsWith("IndexFaces")
                val first = indexing && requests.incrementAndGet() == 1
                val readFailure = target.endsWith("DetectFaces") && reads.incrementAndGet() == 1
                if (first && timeout) Thread.sleep(350)
                val fail = (first && !timeout) || readFailure
                val body = if (fail) """{"__type":"InternalServerError","message":"uncertain"}"""
                    else if (indexing) """{"FaceRecords":[{"Face":{"FaceId":"synthetic-face"}}]}""" else "{}"
                runCatching {
                    it.responseHeaders.add("Content-Type", "application/x-amz-json-1.1")
                    it.sendResponseHeaders(if (fail) 503 else 200, body.length.toLong())
                    it.responseBody.write(body.toByteArray())
                }
            }
        }
        server.start()
        RekognitionClient.builder().region(Region.AP_SOUTHEAST_1)
            .credentialsProvider(StaticCredentialsProvider.create(AwsBasicCredentials.create("synthetic", "synthetic")))
            .endpointOverride(URI("http://127.0.0.1:${server.address.port}"))
            .overrideConfiguration { it.apiCallTimeout(Duration.ofSeconds(3)).apiCallAttemptTimeout(Duration.ofMillis(100)) }
            .build().use { sdk ->
                val client = RekognitionAiClient(sdk, RekognitionProperties(), io.micrometer.core.instrument.simple.SimpleMeterRegistry())
                val outcome = runCatching { client.facesEnroll(byteArrayOf(1), "image/jpeg", "photo.jpg", "photo", null, UUID.randomUUID()) }
                assertAll({ assertIs<AiApiException>(outcome.exceptionOrNull()) }, { assertEquals(1, requests.get()) })
                assertEquals(0, client.facesDetect(byteArrayOf(1), "image/jpeg", "photo.jpg").faces.size)
                assertEquals(2, reads.get(), "An index override must not disable detection retries on the same client")
            }
    }

    private fun server(test: (HttpServer, AtomicInteger) -> Unit) {
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        val executor = Executors.newCachedThreadPool()
        server.executor = executor
        try { test(server, AtomicInteger()) } finally { server.stop(0); executor.shutdownNow() }
    }
}
