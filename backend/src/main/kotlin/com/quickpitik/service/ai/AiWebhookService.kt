package com.quickpitik.service.ai

import com.quickpitik.service.photographer.PhotoBatchIngestService
import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Service

// Off-thread bridge from the webhook controller to the ingest service: the HTTP
// handler verifies + returns 200 fast, while the (network + DB) ingest runs on
// the imageProcessing pool. ingestService.ingest is cross-bean so its
// @Transactional proxy applies, and idempotent so a duplicate delivery is safe.
@Service
class AiWebhookService(
    private val ingestService: PhotoBatchIngestService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("imageProcessing")
    fun onJobEvent(aiJobId: String) {
        runCatching { ingestService.ingest(aiJobId) }
            .onFailure { log.warn("ai webhook ingest failed for job {}: {}", aiJobId, it.message) }
    }
}
