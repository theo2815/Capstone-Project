package com.quickpitik.service.storage

import com.quickpitik.config.StorageProperties
import org.slf4j.LoggerFactory
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.core.env.Environment
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
    private val environment: Environment,
) : StorageService {
    private val log = LoggerFactory.getLogger(javaClass)
    private val root: Path = Paths.get(props.localRoot).toAbsolutePath().also(Files::createDirectories)

    init {
        log.info("LocalFsStorageService active — root={}", root)
        // L-2 — LocalFsStorageService is dev-only. The "presigned" URLs it
        // mints carry expires/method query params but no signature, so any
        // path-knower can read. Loud WARN if STORAGE_BACKEND=LOCAL is active
        // outside an explicitly-dev profile so it's hard to miss in prod
        // startup logs. "default" / empty is the implicit-dev case.
        val activeProfiles = environment.activeProfiles.toList()
        val isDevLike = activeProfiles.isEmpty() ||
            activeProfiles.any { it.equals("dev", ignoreCase = true) || it.equals("default", ignoreCase = true) }
        if (!isDevLike) {
            log.warn(
                "STORAGE_BACKEND=LOCAL with non-dev profile {}. " +
                    "LocalFsStorageService 'presigned' URLs are NOT cryptographically signed — " +
                    "any path-knower can read. Set STORAGE_BACKEND=S3 in production.",
                activeProfiles,
            )
        }
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

    override fun getBytes(key: String): ByteArray = Files.readAllBytes(resolve(key))

    override fun delete(key: String) {
        Files.deleteIfExists(resolve(key))
    }

    override fun exists(key: String): Boolean = Files.exists(resolve(key))

    override fun presignedGetUrl(key: String, ttl: Duration): String =
        buildLocalUrl(key, ttl, "GET")

    override fun presignedDownloadUrl(key: String, ttl: Duration, filename: String): String {
        // StorageDownloadDispositionFilter reads these params off /storage/**
        // and writes the Content-Disposition header — the dev twin of the
        // signed response-content-disposition the S3 impl bakes in.
        val encodedName = URLEncoder.encode(filename, StandardCharsets.UTF_8).replace("+", "%20")
        return buildLocalUrl(key, ttl, "GET") + "&disposition=attachment&filename=$encodedName"
    }

    override fun presignedPutUrl(key: String, ttl: Duration, contentType: String): String =
        buildLocalUrl(key, ttl, "PUT")

    private fun resolve(key: String): Path {
        require(!key.contains("..")) { "key may not traverse: $key" }
        return root.resolve(key.trimStart('/'))
    }

    private fun buildLocalUrl(key: String, ttl: Duration, method: String): String {
        // file:// fallbacks are unloadable by browsers from an http page,
        // and an empty base produces a path-relative URL that resolves
        // against the FE origin (and 404s on Next.js). Default to the
        // dev HTTP mount served by StaticResourceConfig — set
        // STORAGE_PUBLIC_BASE_URL explicitly when running off :8080 or
        // when fronting LocalFs through a CDN/tunnel.
        val base = props.publicBaseUrl?.takeIf { it.isNotBlank() }?.trimEnd('/')
            ?: DEFAULT_DEV_BASE_URL
        // Encode each segment individually so the slashes between
        // events / {id} / cover / {file}.jpg stay literal. Tomcat rejects
        // %2F in path segments (CVE-2007-0450 protection), so encoding the
        // whole key as one string would 400 the request before the static
        // handler runs.
        val encodedPath = key.trimStart('/')
            .split('/')
            .joinToString("/") { segment ->
                URLEncoder.encode(segment, StandardCharsets.UTF_8).replace("+", "%20")
            }
        return "$base/$encodedPath?expires=${System.currentTimeMillis() + ttl.toMillis()}&method=$method"
    }

    private companion object {
        const val DEFAULT_DEV_BASE_URL = "http://localhost:8080/storage"
    }
}
