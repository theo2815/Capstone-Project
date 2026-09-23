package com.quickpitik.service.photographer

import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Component

// Boot-time sanity check: every photographer watermark / cover key the DB
// points at must exist in the configured storage backend. A key that only
// exists on the old backend (e.g. a logo uploaded while STORAGE_BACKEND=LOCAL,
// after a switch to S3/R2) makes EVERY upload by that photographer stick in
// PROCESSING — 32 photos did exactly that on 2026-09-02 before anyone noticed.
// WARN only; the fix is the photographer re-uploading the asset in Settings.
@Component
class PhotographerAssetAuditor(
    private val settingsRepository: PhotographerSettingsRepository,
    private val userSelfieRepository: com.quickpitik.repository.UserSelfieRepository,
    private val storageService: StorageService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("imageProcessing")
    @EventListener(ApplicationReadyEvent::class)
    fun audit() {
        var missing = 0
        var unchecked = 0
        // Only a definite false is missing; a storage error (S3 rethrows every
        // non-404, e.g. a 403 after a token rotation) proves nothing either way.
        fun isMissing(key: String): Boolean = runCatching { storageService.exists(key) }.fold(
            onSuccess = { !it },
            onFailure = {
                unchecked++
                log.warn("Storage asset audit could not check key {}: {}", key, it.toString())
                false
            },
        )
        settingsRepository.findAll().forEach { s ->
            listOf("watermark" to s.watermarkS3Key, "cover" to s.coverS3Key).forEach { (kind, key) ->
                if (key != null && isMissing(key)) {
                    missing++
                    log.warn(
                        "Photographer {} {} key {} is missing from storage — their uploads will stall in PROCESSING until it is re-uploaded",
                        s.userId, kind, key,
                    )
                }
            }
        }
        // Runner selfies: same failure shape — a stored-selfie face search 404s
        // ("Selfie file is no longer available") when the object is missing.
        userSelfieRepository.findAll().forEach { s ->
            if (isMissing(s.s3Key)) {
                missing++
                log.warn(
                    "Runner {} selfie {} key {} is missing from storage — stored-selfie search will 404 until they re-add it",
                    s.userId, s.id, s.s3Key,
                )
            }
        }
        if (missing == 0 && unchecked == 0) log.info("Storage asset audit: all watermark/cover/selfie keys present in storage")
    }
}
