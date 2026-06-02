package com.quickpitik.common

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IndexingMode
import com.quickpitik.service.ai.AiApiClient
import org.slf4j.LoggerFactory
import org.springframework.boot.ApplicationArguments
import org.springframework.boot.ApplicationRunner
import org.springframework.stereotype.Component

// Registers this backend's ai-webhook callback URL with ai-api on startup so job
// completions are pushed instead of polled. Prod-only: ai-api's SSRF guard
// rejects private IPs, so a localhost dev backend can't be a target (dev leaves
// webhook.enabled=false and relies on the poll backstop). Idempotent +
// best-effort — a duplicate registration is harmless because ingest is idempotent.
@Component
class AiWebhookSubscriptionRunner(
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
) : ApplicationRunner {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun run(args: ApplicationArguments?) {
        val wh = aiApiProperties.webhook
        if (!aiApiProperties.enabled || aiApiProperties.indexingMode != IndexingMode.BATCH || !wh.enabled) {
            log.info(
                "ai webhook subscription skipped (ai-api enabled={}, mode={}, webhook.enabled={})",
                aiApiProperties.enabled, aiApiProperties.indexingMode, wh.enabled,
            )
            return
        }
        if (wh.publicUrl.isBlank() || wh.hmacSecret.isBlank()) {
            log.warn("ai webhook subscription skipped — AI_WEBHOOK_PUBLIC_URL or AI_WEBHOOK_HMAC_SECRET is empty")
            return
        }
        val callbackUrl = "${wh.publicUrl.trimEnd('/')}/api/v1/internal/ai-webhooks"
        try {
            if (aiApiClient.listWebhookUrls().any { it == callbackUrl }) {
                log.info("ai webhook already registered: {}", callbackUrl)
                return
            }
            val ok = aiApiClient.registerWebhook(callbackUrl, listOf("job.completed", "job.failed"), wh.hmacSecret)
            if (ok) {
                log.info("Registered ai webhook -> {}", callbackUrl)
            } else {
                log.warn("ai webhook registration returned failure for {}", callbackUrl)
            }
        } catch (ex: Exception) {
            log.warn("ai webhook registration failed for {}: {}", callbackUrl, ex.message)
        }
    }
}
