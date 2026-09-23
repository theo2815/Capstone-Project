package com.quickpitik.service.photographer

import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component

@Component
class PhotoUploadCleanupTrigger(
    private val photoUploadService: PhotoUploadService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(fixedDelayString = "\${app.storage.direct-upload.cleanup-interval:PT1M}")
    fun cleanup() {
        try {
            photoUploadService.cleanupDirectUploads()
        } catch (error: Exception) {
            log.warn("Upload cleanup sweep failed error={}", error.javaClass.simpleName)
        }
    }
}
