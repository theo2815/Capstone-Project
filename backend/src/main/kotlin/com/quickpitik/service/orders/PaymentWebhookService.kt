package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.orders.PaymentWebhookRequest
import com.quickpitik.dto.orders.PaymentWebhookResponse
import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

/**
 * Provider-stub webhook handler. Real GCash/Maya/card integration deferred —
 * for v1 this lets ops manually drive a PENDING order to PAID via curl. HMAC
 * verification (real-world `X-QuickPitik-Signature`) is a placeholder TODO; the
 * webhook is allow-listed as public in SecurityConfig for the demo.
 *
 * Idempotency: providerRef is unique per (provider, providerRef) so retried
 * webhooks no-op cleanly without double-minting grants.
 */
@Service
@Transactional
class PaymentWebhookService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val paymentRepository: PaymentRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
) {
    fun handle(provider: String, request: PaymentWebhookRequest): PaymentWebhookResponse {
        if (provider.isBlank()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "provider is required",
                field = "provider",
            )
        }
        if (request.providerRef.isBlank()) {
            throw ValidationException(
                code = ErrorCodes.MISSING_PAYMENT_REFERENCE,
                message = "providerRef is required",
                field = "providerRef",
            )
        }

        val order = orderRepository.findById(request.orderId).orElseThrow {
            NotFoundException(code = ErrorCodes.ORDER_NOT_FOUND, message = "Order not found")
        }

        val existing = paymentRepository.findByProviderAndProviderRef(provider, request.providerRef)
        if (existing != null && existing.orderId != order.id) {
            throw ConflictException(
                code = ErrorCodes.ORDER_DUPLICATE_PAYMENT,
                message = "providerRef already used by another order",
            )
        }

        val now = OffsetDateTime.now()
        val grantsBefore = downloadGrantRepository.findByIdOrderId(order.id).size

        if (existing == null) {
            paymentRepository.save(
                Payment(
                    orderId = order.id,
                    provider = provider,
                    providerRef = request.providerRef,
                    amountPhp = request.amountPhp ?: order.totalPhp,
                    status = mapWebhookStatus(request.status),
                    paidAt = if (request.status.equals("SUCCEEDED", ignoreCase = true)) now else null,
                ),
            )
        }

        if (request.status.equals("SUCCEEDED", ignoreCase = true) && order.status != OrderStatus.PAID &&
            order.status != OrderStatus.FULFILLED
        ) {
            order.status = OrderStatus.PAID
            order.paidAt = now
            orderRepository.save(order)
            mintGrantsIfMissing(order.id, now)
        }

        val grantsAfter = downloadGrantRepository.findByIdOrderId(order.id).size
        return PaymentWebhookResponse(
            orderId = order.id,
            orderStatus = order.status.name,
            grantsMinted = grantsAfter - grantsBefore,
        )
    }

    private fun mintGrantsIfMissing(orderId: UUID, now: OffsetDateTime) {
        val existing = downloadGrantRepository.findByIdOrderId(orderId).map { it.id.photoId }.toSet()
        val photos = orderItemRepository.findByIdOrderId(orderId).map { it.id.photoId }
        val grantedUntil = now.plusYears(1)
        photos.filter { it !in existing }.forEach { photoId ->
            downloadGrantRepository.save(
                DownloadGrant(
                    id = DownloadGrantId(orderId = orderId, photoId = photoId),
                    grantedUntil = grantedUntil,
                ),
            )
        }
    }

    private fun mapWebhookStatus(raw: String): PaymentStatus =
        when (raw.uppercase()) {
            "SUCCEEDED", "SUCCESS", "PAID" -> PaymentStatus.SUCCEEDED
            "FAILED", "DECLINED", "ERROR" -> PaymentStatus.FAILED
            else -> PaymentStatus.PENDING
        }
}
