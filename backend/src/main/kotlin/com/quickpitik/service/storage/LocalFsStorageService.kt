package com.quickpitik.service.storage

import com.quickpitik.config.StorageProperties
import org.slf4j.LoggerFactory
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.stereotype.Service
import java.io.InputStream
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.time.Duration

@Service
@ConditionalOnProperty(prefix = "app.storage", name = ["backend"], havingValue = "LOCAL", matchIfMissing = true)
class LocalFsStorageService(
    private val props: StorageProperties,
) : StorageService {
    private val log = LoggerFactory.getLogger(javaClass)
    private val root: Path = Paths.get(props.localRoot).toAbsolutePath().also(Files::createDirectories)

    init {
        log.info("LocalFsStorageService active — root={}", root)
    }

    override fun put(key: String, bytes: ByteArray, contentType: String): StoredObject {
        val target = resolve(key)
        Files.createDirectories(target.parent)
        Files.write(target, bytes)
        return StoredObject(key = key, sizeBytes = bytes.size.toLong(), contentType = contentType)
    }

    override fun put(key: String, stream: InputStream, contentLength: Long, contentType: String): StoredObject {
        val target = resolve(key)
        Files.createDirectories(target.parent)
        stream.use { Files.copy(it, target, StandardCopyOption.REPLACE_EXISTING) }
        return StoredObject(key = key, sizeBytes = Files.size(target), contentType = contentType)
    }

    override fun delete(key: String) {
        Files.deleteIfExists(resolve(key))
    }

    override fun exists(key: String): Boolean = Files.exists(resolve(key))

    override fun presignedGetUrl(key: String, ttl: Duration): String =
        buildLocalUrl(key, ttl, "GET")

    override fun presignedPutUrl(key: String, ttl: Duration, contentType: String): String =
        buildLocalUrl(key, ttl, "PUT")

    private fun resolve(key: String): Path {
        require(!key.contains("..")) { "key may not traverse: $key" }
        return root.resolve(key.trimStart('/'))
    }

    private fun buildLocalUrl(key: String, ttl: Duration, method: String): String {
        val base = props.publicBaseUrl?.trimEnd('/') ?: "file://${root.toUri().path.trimEnd('/')}"
        val encoded = URLEncoder.encode(key, StandardCharsets.UTF_8).replace("+", "%20")
        return "$base/$encoded?expires=${System.currentTimeMillis() + ttl.toMillis()}&method=$method"
    }
}
