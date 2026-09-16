package com.quickpitik.service.disbursement

import com.quickpitik.dto.orders.PaymongoTransferWebhookEvent
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.ProcessedWebhookEvent
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.ProcessedWebhookEventRepository
import com.quickpitik.service.earnings.PayoutCycleTransitions
import com.quickpitik.service.earnings.PayoutTransition
import io.micrometer.core.instrument.MeterRegistry
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime

/**
 * Applies PayMongo transfer webhook events to payout cycles: the provider — not
 * the admin — is the source of truth for whether money actually moved.
 *   transfer.outward.successful → PROCESSING → PAID
 *   transfer.outward.failed     → PROCESSING → FAILED   (reservation releases)
 *   transfer.outward.returned   → PROCESSING/PAID → RETURNED
 *
 * Deduped after applying the outcome in the same transaction. Unknown transfers
 * and conflicting outcomes return 503 for redelivery, never a permanent skip.
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
    private val payoutCycleTransitions: PayoutCycleTransitions,
    private val meterRegistry: MeterRegistry,
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

        val transition = transitionFor(type)
            ?: run {
                log.info("PayMongo transfer webhook {}: unhandled type {} — acknowledged", eventId, type)
                return mapOf("acknowledged" to type)
            }

        val cycle = payoutCycleRepository.findByProviderTransferId(transferId)
        if (cycle == null) {
            // The callback can beat the transaction storing the transfer id.
            throw retryLater("Transfer has not been linked to a payout yet")
        }
        if (cycle.provider != PROVIDER) throw retryLater("Transfer provider does not match the payout")

        val result = applyTransition(cycle, transition)
        // Record the event as handled only once we've acted on it (transition
        // applied, or a definitive skip). Ignored/acknowledged no-ops above are
        // left unrecorded — re-delivering them just re-no-ops harmlessly.
        processedWebhookEventRepository.save(ProcessedWebhookEvent(provider = PROVIDER, eventId = eventId))
        return result
    }

    private fun applyTransition(cycle: PayoutCycle, transition: PayoutTransition): Map<String, Any?> {
        val from = cycle.status
        val target = transition.to
        if (from == target || (from == PayoutCycleStatus.RETURNED && target == PayoutCycleStatus.PAID)) {
            // A confirmed return supersedes a successful event delivered later.
            return mapOf("skipped" to "already-applied", "cycle" to cycle.id, "from" to from.wire)
        }
        // The allowed-from set is the shared table's, but the response is not:
        // an out-of-order delivery must ask PayMongo to redeliver (503), never
        // hand it the admin surface's 409.
        if (from !in transition.allowedFrom) {
            // Redelivered forever until reconciled by hand — alert on this counter.
            meterRegistry.counter("qp.payout.webhook.conflict", "from", from.wire, "to", target.wire).increment()
            log.warn(
                "PayMongo transfer webhook: cycle {} is {}, cannot move to {} — reconciliation required",
                cycle.id, from.wire, target.wire,
            )
            throw retryLater("Payout outcome conflicts with its current state; reconciliation required")
        }
        when (target) {
            PayoutCycleStatus.PAID -> cycle.settledAt = OffsetDateTime.now()
            PayoutCycleStatus.FAILED -> cycle.failedReason = "PayMongo reported the transfer failed"
            PayoutCycleStatus.RETURNED -> cycle.returnedAt = OffsetDateTime.now()
            else -> {}
        }
        payoutCycleTransitions.apply(cycle, transition, actorId = null, reason = "paymongo webhook")
        return mapOf("ok" to true, "cycle" to cycle.id, "status" to target.wire)
    }

    private fun retryLater(message: String) = ApiException(
        HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.INVALID_STATE_TRANSITION, message, retryAfterSeconds = 30,
    )

    private fun transitionFor(type: String): PayoutTransition? = when (type) {
        // ⚠ VERIFY-IN-TEST-MODE: confirm these exact event strings.
        "transfer.outward.successful" -> PayoutTransition.ProviderPaid
        "transfer.outward.failed" -> PayoutTransition.ProviderFailed
        "transfer.outward.returned" -> PayoutTransition.ProviderReturned
        else -> null
    }

    private companion object {
        const val PROVIDER = "paymongo"
    }
}
