package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.orders.RefundDisputeDto
import com.quickpitik.dto.orders.RefundRequest
import com.quickpitik.dto.orders.RefundResponse
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeReason
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.websocket.AdminInboxEvent
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class RefundService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val disputeRepository: DisputeRepository,
    private val eventPublisher: ApplicationEventPublisher,
) {
    /**
     * POST /api/v1/me/orders/{id}/refund.
     *
     * Creates one Dispute row per photoId, all sharing the same reason + note.
     * The unique partial index on (order_id, photo_id) WHERE status IN
     * ('open', 'escalated') prevents duplicate open disputes per photo per the
     * refund-helpers.ts canSubmitRefund rule.
     *
     * Sales receipt is unaffected — the order keeps its PAID status until an
     * admin resolves the dispute (Phase G).
     */
    fun submit(userId: UUID, orderId: UUID, request: RefundRequest): RefundResponse {
        if (request.photoIds.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "photoIds must not be empty",
                field = "photoIds",
            )
        }

        val order = orderRepository.findById(orderId).orElseThrow {
            NotFoundException(code = ErrorCodes.ORDER_NOT_FOUND, message = "Order not found")
        }
        if (order.userId != userId) {
            throw NotFoundException(code = ErrorCodes.ORDER_NOT_FOUND, message = "Order not found")
        }

        val reason = DisputeReason.fromWire(request.reason)
        val note = request.note.trim()

        val orderPhotoIds = orderItemRepository.findByIdOrderId(orderId)
            .map { it.id.photoId }
            .toSet()
        val invalid = request.photoIds.toSet() - orderPhotoIds
        if (invalid.isNotEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "Photos not in order: ${invalid.joinToString()}",
                field = "photoIds",
            )
        }

        val now = OffsetDateTime.now()
        val createdDisputes = mutableListOf<Dispute>()
        for (photoId in request.photoIds.distinct()) {
            val existing = disputeRepository.findOpenForOrderPhoto(orderId, photoId)
            if (existing != null) {
                throw ConflictException(
                    code = ErrorCodes.CONFLICT,
                    message = "A refund request for photo $photoId is already open",
                )
            }
            try {
                val saved = disputeRepository.save(
                    Dispute(
                        orderId = orderId,
                        photoId = photoId,
                        runnerId = userId,
                        reasonWire = reason.wire,
                        note = note,
                        openedAt = now,
                    ),
                )
                createdDisputes.add(saved)
            } catch (ex: DataIntegrityViolationException) {
                // Race with the partial unique index — surface as a clean conflict.
                throw ConflictException(
                    code = ErrorCodes.CONFLICT,
                    message = "A refund request for photo $photoId is already open",
                )
            }
        }

        // Push one AdminInboxEvent per created dispute so the admin queue
        // updates in real time without a refresh. AFTER_COMMIT only —
        // rollback discards both the rows AND the would-be pushes.
        createdDisputes.forEach { dispute ->
            eventPublisher.publishEvent(
                AdminInboxEvent(
                    payload = mapOf(
                        "type" to "dispute_filed",
                        "entityId" to dispute.id.toString(),
                        "actorId" to userId.toString(),
                        "occurredAt" to dispute.openedAt.toString(),
                    ),
                ),
            )
        }

        return RefundResponse(
            orderId = orderId,
            disputes = createdDisputes.map {
                RefundDisputeDto(
                    id = it.id,
                    photoId = it.photoId,
                    status = it.statusWire,
                    openedAt = it.openedAt,
                )
            },
        )
    }
}
