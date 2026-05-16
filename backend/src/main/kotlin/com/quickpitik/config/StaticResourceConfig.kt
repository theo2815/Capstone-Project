package com.quickpitik.config

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.context.annotation.Configuration
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer
import java.nio.file.Files
import java.nio.file.Paths

// Dev-only HTTP mount for the LocalFs storage tree. Mirrors what S3 +
// CloudFront do in prod: maps the "presigned" GET URLs minted by
// LocalFsStorageService back to bytes on disk. Without this the URL is
// http://localhost:8080/storage/... but no handler answers and every <img>
// in the FE paints the broken-image glyph. Conditional on
// app.storage.backend=LOCAL so the prod S3StorageService never exposes the
// disk path.
@Configuration
@ConditionalOnProperty(prefix = "app.storage", name = ["backend"], havingValue = "LOCAL", matchIfMissing = true)
class StaticResourceConfig(
    private val storageProperties: StorageProperties,
) : WebMvcConfigurer {
    override fun addResourceHandlers(registry: ResourceHandlerRegistry) {
        val root = Paths.get(storageProperties.localRoot).toAbsolutePath()
        Files.createDirectories(root)
        val location = root.toUri().toString()
        val withTrailing = if (location.endsWith("/")) location else "$location/"
        registry.addResourceHandler("/storage/**").addResourceLocations(withTrailing)
    }
}
