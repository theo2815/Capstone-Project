package com.quickpitik.service.disbursement

import com.quickpitik.dto.orders.PaymongoTransferWebhookEvent
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.ProcessedWebhookEvent
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.ProcessedWebhookEventRepository
import com.quickpitik.service.earnings.PayoutStatusRecorder
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime

/**
 * Applies PayMongo transfer webhook events to payout cycles: the provider — not
 * the admin — is the source of truth for whether money actually moved.
 *   transfer.outward.successful → PROCESSING → PAID
 *   transfer.outward.failed     → PROCESSING → FAILED   (reservation releases)
 *   transfer.outward.returned   → PAID       → RETURNED
 *
 * Deduped by event id (insert-first) so a redelivery is a no-op. A transition
 * whose source state doesn't match is logged and skipped (already applied /
 * out-of-order) rather than 409'd, so PayMongo doesn't retry forever.
 *
 * ⚠ Inactive until PayMongo is enabled + verified (docs §I): the exact event
 * strings are provisional, and photographer inbox notification on webhook-driven
 * outcomes is owed (the existing pushMessage path is admin-actor).
 */
@Service
@Transactional
class PayoutDisbursementWebhookService(
    private val payoutCycleRepository: PayoutCycleRepository,
    private val processedWebhookEventRepository: ProcessedWebhookEventRepository,
    private val payoutStatusRecorder: PayoutStatusRecorder,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun handle(event: PaymongoTransferWebhookEvent): Map<String, Any?> {
        val eventId = event.data?.id
        if (eventId.isNullOrBlank()) {
            log.warn("PayMongo transfer webhook: missing event id — ignoring")
            return mapOf("ignored" to "no-event-id")
        }
        // Dedupe (common case, clean): a redelivery of an already-handled event
        // short-circuits before any work. The rare concurrent duplicate is caught
        // by the UNIQUE at markProcessed() below — that write fails, the whole TX
        // (transition included) rolls back, and PayMongo's retry hits this check.
        if (processedWebhookEventRepository.existsByProviderAndEventId(PROVIDER, eventId)) {
            return mapOf("deduped" to true)
        }

        val type = event.data.attributes?.type
        val transferId = event.data.attributes?.data?.id
        if (type.isNullOrBlank() || transferId.isNullOrBlank()) {
            log.warn("PayMongo transfer webhook {}: missing type/transfer id — ignoring", eventId)
            return mapOf("ignored" to "no-type-or-transfer")
        }

        val target = targetStatusFor(type)
            ?: run {
                log.info("PayMongo transfer webhook {}: unhandled type {} — acknowledged", eventId, type)
                return mapOf("acknowledged" to type)
            }

        val cycle = payoutCycleRepository.findByProviderTransferId(transferId)
        if (cycle == null) {
            log.warn("PayMongo transfer webhook {}: no cycle for transfer {} — ignoring", eventId, transferId)
            return mapOf("ignored" to "unknown-transfer")
        }

        val result = applyTransition(cycle, target)
        // Record the event as handled only once we've acted on it (transition
        // applied, or a definitive skip). Ignored/acknowledged no-ops above are
        // left unrecorded — re-delivering them just re-no-ops harmlessly.
        processedWebhookEventRepository.save(ProcessedWebhookEvent(provider = PROVIDER, eventId = eventId))
        return result
    }

    private fun applyTransition(cycle: PayoutCycle, target: PayoutCycleStatus): Map<String, Any?> {
        val from = cycle.status
        val validFrom = when (target) {
            PayoutCycleStatus.PAID, PayoutCycleStatus.FAILED -> from == PayoutCycleStatus.PROCESSING
            PayoutCycleStatus.RETURNED -> from == PayoutCycleStatus.PAID
            else -> false
        }
        if (!validFrom) {
            // Out-of-order or duplicate outcome — the state already moved on.
            log.warn(
                "PayMongo transfer webhook: cycle {} is {}, cannot move to {} — skipping",
                cycle.id, from.wire, target.wire,
            )
            return mapOf("skipped" to "invalid-from", "cycle" to cycle.id, "from" to from.wire)
        }
        cycle.status = target
        when (target) {
            PayoutCycleStatus.PAID -> cycle.settledAt = OffsetDateTime.now()
            PayoutCycleStatus.FAILED -> cycle.failedReason = "PayMongo reported the transfer failed"
            PayoutCycleStatus.RETURNED -> cycle.returnedAt = OffsetDateTime.now()
            else -> {}
        }
        payoutCycleRepository.save(cycle)
        payoutStatusRecorder.record(cycle.id, from, target, "provider", null, "paymongo webhook")
        return mapOf("ok" to true, "cycle" to cycle.id, "status" to target.wire)
    }

    private fun targetStatusFor(type: String): PayoutCycleStatus? = when (type) {
        // ⚠ VERIFY-IN-TEST-MODE: confirm these exact event strings.
        "transfer.outward.successful" -> PayoutCycleStatus.PAID
        "transfer.outward.failed" -> PayoutCycleStatus.FAILED
        "transfer.outward.returned" -> PayoutCycleStatus.RETURNED
        else -> null
    }

    private companion object {
        const val PROVIDER = "paymongo"
    }
}
