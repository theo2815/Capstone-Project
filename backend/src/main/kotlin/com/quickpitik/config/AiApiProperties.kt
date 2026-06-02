package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

@ConfigurationProperties(prefix = "app.ai-api")
data class AiApiProperties(
    // Master switch — when false, all server-side ai-api calls are skipped.
    // Photo upload still succeeds (faces + bibs are best-effort already), runner
    // selfie upload skips the quality gate, runner face-search short-circuits to
    // 503 AI_API_UNAVAILABLE. Flip to true via AI_API_ENABLED=true when ai-api
    // is running and you want face/bib indexing + search to work.
    val enabled: Boolean = true,
    val baseUrl: String = "http://localhost:8000",
    val apiKey: String = "dev-only-ai-api-key-DO-NOT-USE-IN-PRODUCTION",
    val connectTimeout: Duration = Duration.ofSeconds(5),
    val readTimeout: Duration = Duration.ofSeconds(30),
    val maxRetries: Int = 3,
    val backoffBaseMillis: Long = 500L,
    val faceMatchThresholdDefault: Double = 0.6,
    val faceTopKDefault: Int = 5,
    val bibConfidenceThresholdDefault: Double = 0.7,
    // Async indexing reconciliation sweep (PhotoIndexingTrigger.reconcile):
    // how often it runs, and the max attempts before a photo settles as FAILED.
    val reconcileIntervalMs: Long = 60_000L,
    val maxIndexingAttempts: Int = 5,
    // HTTP connection pool to ai-api. Async indexing issues many concurrent
    // calls (face + bib per photo across the imageProcessing pool), so a pool
    // replaces the old one-connection-per-request factory.
    val maxConnections: Int = 50,
    val maxConnectionsPerRoute: Int = 20,
    // Phase C bulk indexing. PER_PHOTO (default) is the Phase-B behaviour:
    // one enroll + one bib HTTP call per photo, off the upload request. BATCH
    // stands that down and drains PENDING photos per event into mega jobs +
    // webhook/poll ingest. Inert until flipped via INDEXING_MODE=batch.
    val indexingMode: IndexingMode = IndexingMode.PER_PHOTO,
    val batch: BatchIndexingProperties = BatchIndexingProperties(),
    val webhook: AiWebhookProperties = AiWebhookProperties(),
)

enum class IndexingMode {
    PER_PHOTO,
    BATCH,
}

data class BatchIndexingProperties(
    // Photos per mega job. Bounds memory: the drain reads this many ORIGINAL
    // images into memory to build the multipart upload. 50 = one ai-api chunk.
    // ai-api caps mega at 500; raise with care (memory scales linearly).
    val maxSize: Int = 50,
    val drainIntervalMs: Long = 30_000L,
    // Skip photos the upload just landed (give trickle uploads a beat to settle
    // before sweeping them into a batch).
    val drainGraceSeconds: Long = 60L,
    // How many distinct events to drain per tick (each is one batch = 2 jobs).
    val maxEventsPerTick: Int = 5,
)

data class AiWebhookProperties(
    // Webhook receiver is a prod-only latency optimisation: ai-api's SSRF guard
    // rejects private IPs at registration, so a localhost dev backend can't be a
    // webhook target. Dev runs poll-only (the poll backstop does the real work).
    val enabled: Boolean = false,
    // The externally-reachable backend origin ai-api POSTs callbacks to.
    val publicUrl: String = "",
    // Shared secret registered with ai-api + used to verify inbound signatures.
    val hmacSecret: String = "",
    val pollIntervalMs: Long = 30_000L,
    // Only poll jobs that have been SUBMITTED longer than this (give the webhook
    // a chance to arrive first in prod; in dev this is just the ingest cadence).
    val pollGraceSeconds: Long = 90L,
)
