package com.quickpitik.service.storage

import com.quickpitik.config.StorageProperties
import org.slf4j.LoggerFactory
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.stereotype.Service
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider
import software.amazon.awssdk.core.sync.RequestBody
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.S3Configuration
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest
import software.amazon.awssdk.services.s3.model.GetObjectRequest
import software.amazon.awssdk.services.s3.model.HeadObjectRequest
import software.amazon.awssdk.services.s3.model.NoSuchKeyException
import software.amazon.awssdk.services.s3.model.PutObjectRequest
import software.amazon.awssdk.services.s3.presigner.S3Presigner
import software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest
import java.io.InputStream
import java.net.URI
import java.time.Duration

@Service
@ConditionalOnProperty(prefix = "app.storage", name = ["backend"], havingValue = "S3")
class S3StorageService(
    private val props: StorageProperties,
) : StorageService, AutoCloseable {
    private val log = LoggerFactory.getLogger(javaClass)
    private val client: S3Client = buildClient()
    private val presigner: S3Presigner = buildPresigner()

    init {
        log.info("S3StorageService active — bucket={} region={} endpoint={}", props.bucket, props.region, props.endpoint)
    }

    override fun put(key: String, bytes: ByteArray, contentType: String): StoredObject {
        client.putObject(
            PutObjectRequest.builder().bucket(props.bucket).key(key).contentType(contentType).build(),
            RequestBody.fromBytes(bytes),
        )
        return StoredObject(key = key, sizeBytes = bytes.size.toLong(), contentType = contentType)
    }

    override fun put(key: String, stream: InputStream, contentLength: Long, contentType: String): StoredObject {
        client.putObject(
            PutObjectRequest.builder().bucket(props.bucket).key(key).contentType(contentType).build(),
            RequestBody.fromInputStream(stream, contentLength),
        )
        return StoredObject(key = key, sizeBytes = contentLength, contentType = contentType)
    }

    override fun delete(key: String) {
        client.deleteObject(DeleteObjectRequest.builder().bucket(props.bucket).key(key).build())
    }

    override fun exists(key: String): Boolean = try {
        client.headObject(HeadObjectRequest.builder().bucket(props.bucket).key(key).build())
        true
    } catch (_: NoSuchKeyException) {
        false
    }

    override fun presignedGetUrl(key: String, ttl: Duration): String {
        val req = GetObjectRequest.builder().bucket(props.bucket).key(key).build()
        val presignRequest = GetObjectPresignRequest.builder().signatureDuration(ttl).getObjectRequest(req).build()
        return presigner.presignGetObject(presignRequest).url().toExternalForm()
    }

    override fun presignedPutUrl(key: String, ttl: Duration, contentType: String): String {
        val req = PutObjectRequest.builder().bucket(props.bucket).key(key).contentType(contentType).build()
        val presignRequest = PutObjectPresignRequest.builder().signatureDuration(ttl).putObjectRequest(req).build()
        return presigner.presignPutObject(presignRequest).url().toExternalForm()
    }

    override fun close() {
        client.close()
        presigner.close()
    }

    private fun buildClient(): S3Client {
        val builder = S3Client.builder()
            .region(Region.of(props.region))
            .credentialsProvider(credentials())
            .serviceConfiguration(
                S3Configuration.builder().pathStyleAccessEnabled(props.pathStyleAccess).build(),
            )
        props.endpoint?.let { builder.endpointOverride(URI.create(it)) }
        return builder.build()
    }

    private fun buildPresigner(): S3Presigner {
        val builder = S3Presigner.builder()
            .region(Region.of(props.region))
            .credentialsProvider(credentials())
        props.endpoint?.let { builder.endpointOverride(URI.create(it)) }
        return builder.build()
    }

    private fun credentials() =
        if (props.accessKey != null && props.secretKey != null) {
            StaticCredentialsProvider.create(AwsBasicCredentials.create(props.accessKey, props.secretKey))
        } else {
            DefaultCredentialsProvider.create()
        }
}
