package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.orders.CreateOrderItem
import com.quickpitik.dto.orders.CreateOrderRequest
import com.quickpitik.dto.orders.OrderDetailDto
import com.quickpitik.dto.orders.OrderListItemDto
import com.quickpitik.dto.orders.OrderPhotoDetailDto
import com.quickpitik.dto.orders.OrderResponse
import com.quickpitik.dto.orders.OrderResponseItem
import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentMethod
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.entity.Photo
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.CartItemRepository
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.earnings.TransactionMintingService
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID

@Service
@Transactional
class OrderService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val paymentRepository: PaymentRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val photoRepository: PhotoRepository,
    private val eventRepository: EventRepository,
    private val userRepository: UserRepository,
    private val cartItemRepository: CartItemRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val transactionMintingService: TransactionMintingService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    /**
     * POST /api/v1/orders.
     *
     * Splits cart items by event so each Order row stays single-event (matches the
     * FE's MockOrder.eventId shape and keeps refund attribution unambiguous). All
     * created rows share the same idempotency_key so a network retry produces no
     * duplicate charge — the unique index on (idempotency_key, event_id) enforces
     * this at the DB level. The idempotency_key arrives via the Idempotency-Key
     * HTTP header (RFC 9110 §9.2.2) — the controller validates + parses it before
     * this method runs, so we trust the parameter here.
     *
     * Real GCash/Maya/card integration is out of scope for v1. Orders are created
     * in PAID state immediately and download_grants are minted in the same TX so
     * the FE can show success + downloads without waiting for a webhook.
     */
    fun create(userId: UUID?, request: CreateOrderRequest, idempotencyKey: String): OrderResponse {
        if (request.items.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "items must not be empty",
                field = "items",
            )
        }

        val paymentMethod = PaymentMethod.fromWire(request.paymentMethod)

        val recipientEmail = resolveRecipientEmail(userId, request.recipientEmail)

        // Idempotency: a re-POST with the same key returns the existing order set
        // (HTTP 200, not 409 — Q-008). Locked at the DB unique index too.
        val existing = orderRepository.findByIdempotencyKey(idempotencyKey)
        if (existing.isNotEmpty()) {
            return toOrderResponse(pickPrimary(existing))
        }

        val photos = loadAndValidatePhotos(request.items)
        val groupedByEvent = request.items.groupBy { it.eventId }
        val eventLookup = eventRepository.findAllById(groupedByEvent.keys).associateBy { it.id }
        for (eventId in groupedByEvent.keys) {
            val event = eventLookup[eventId]
                ?: throw NotFoundException(
                    code = ErrorCodes.EVENT_NOT_FOUND,
                    message = "Event not found",
                )
            if (event.status == EventStatus.ARCHIVED) {
                throw ConflictException(
                    code = ErrorCodes.EVENT_ARCHIVED,
                    message = "Event ${event.slug} is archived",
                )
            }
        }

        val created = mutableListOf<Order>()
        val now = OffsetDateTime.now()
        try {
            for ((eventId, items) in groupedByEvent) {
                val totalPhp = items.fold(BigDecimal.ZERO) { acc, it ->
                    acc + (photos[it.photoId]?.pricePhp ?: BigDecimal.ZERO)
                }
                val order = orderRepository.save(
                    Order(
                        userId = userId,
                        eventId = eventId,
                        recipientEmail = recipientEmail,
                        paymentMethodWire = paymentMethod.wire,
                        status = OrderStatus.PAID,
                        totalPhp = totalPhp,
                        idempotencyKey = idempotencyKey,
                        paidAt = now,
                    ),
                )
                items.forEach { item ->
                    val photo = photos.getValue(item.photoId)
                    orderItemRepository.save(
                        OrderItem(
                            id = OrderItemId(orderId = order.id, photoId = item.photoId),
                            pricePhpAtPurchase = photo.pricePhp,
                        ),
                    )
                }
                paymentRepository.save(
                    Payment(
                        orderId = order.id,
                        provider = paymentMethod.wire,
                        providerRef = "stub-${order.id}",
                        amountPhp = totalPhp,
                        status = PaymentStatus.SUCCEEDED,
                        paidAt = now,
                    ),
                )
                mintDownloadGrants(order.id, items.map { it.photoId }, now)
                created.add(order)
            }
        } catch (ex: DataIntegrityViolationException) {
            // Concurrent retry hit the unique (idempotency_key, event_id) index.
            // Re-read existing rows and return them.
            val replay = orderRepository.findByIdempotencyKey(idempotencyKey)
            if (replay.isNotEmpty()) {
                return toOrderResponse(pickPrimary(replay))
            }
            throw ex
        }

        // Mint earnings rows in the same TX so /me/photographer/earnings sees
        // the sale immediately. Idempotent at the (order_id, photo_id) unique
        // index — replays are clean no-ops.
        created.forEach { transactionMintingService.mintForPaidOrder(it.id) }

        // Best-effort: clear the items the runner just bought from their cart.
        // Not transactional with order creation — if it fails the order still stands.
        // N-1 — log on failure so the "item stuck in cart after purchase" bug
        // is debuggable without re-running the request under a profiler.
        if (userId != null) {
            request.items.forEach { item ->
                runCatching {
                    cartItemRepository.deleteByUserIdAndPhotoId(userId, item.photoId)
                }.onFailure { ex ->
                    log.warn(
                        "cart-clear failed for user={} photo={}: {}",
                        userId,
                        item.photoId,
                        ex.message,
                    )
                }
            }
        }

        return toOrderResponse(pickPrimary(created))
    }

    @Transactional(readOnly = true)
    fun listForUser(userId: UUID, params: PaginationParams): PaginatedResponse<OrderListItemDto> {
        val page = orderRepository.findByUserIdOrderByPaidAtDescCreatedAtDesc(
            userId = userId,
            pageable = OffsetLimitPageable(params),
        )
        if (page.isEmpty) return PaginatedResponse.empty(params)
        val items = hydrateList(page.content)
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    @Transactional(readOnly = true)
    fun getDetail(userId: UUID, orderId: UUID): OrderDetailDto {
        val order = orderRepository.findById(orderId).orElseThrow {
            NotFoundException(code = ErrorCodes.ORDER_NOT_FOUND, message = "Order not found")
        }
        if (order.userId != userId) {
            // Anti-IDOR — never reveal that the order exists for another user.
            throw NotFoundException(code = ErrorCodes.ORDER_NOT_FOUND, message = "Order not found")
        }
        return hydrateDetail(order)
    }

    private fun resolveRecipientEmail(userId: UUID?, requestEmail: String?): String {
        if (userId != null) {
            val user = userRepository.findById(userId).orElseThrow {
                UnauthorizedException(code = ErrorCodes.UNAUTHORIZED, message = "User not found")
            }
            return user.email
        }
        // M-4 — Bean Validation @Email on the controller DTO covers the canonical
        // path; this is service-boundary defense-in-depth in case a future
        // controller path bypasses the annotation. Lowercase + trim before
        // storing so receipt deduplication doesn't fork on Aa@b.com vs aa@b.com.
        val email = requestEmail?.trim()?.lowercase().orEmpty()
        if (email.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "recipientEmail is required for guest checkout",
                field = "recipientEmail",
            )
        }
        if (!EMAIL_REGEX.matches(email)) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "recipientEmail must be a valid email address",
                field = "recipientEmail",
            )
        }
        return email
    }

    private fun loadAndValidatePhotos(items: List<CreateOrderItem>): Map<UUID, Photo> {
        val photoIds = items.map { it.photoId }.toSet()
        val photos = photoRepository.findAllById(photoIds).associateBy { it.id }
        if (photos.size != photoIds.size) {
            throw NotFoundException(
                code = ErrorCodes.PHOTO_NOT_FOUND,
                message = "One or more photos not found",
            )
        }
        items.forEach { item ->
            val photo = photos.getValue(item.photoId)
            if (photo.eventId != item.eventId) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "Photo ${item.photoId} does not belong to event ${item.eventId}",
                    field = "items",
                )
            }
        }
        return photos
    }

    private fun mintDownloadGrants(orderId: UUID, photoIds: List<UUID>, now: OffsetDateTime) {
        val grantedUntil = now.plusYears(1)
        photoIds.forEach { photoId ->
            downloadGrantRepository.save(
                DownloadGrant(
                    id = DownloadGrantId(orderId = orderId, photoId = photoId),
                    grantedUntil = grantedUntil,
                ),
            )
        }
    }

    private fun pickPrimary(orders: List<Order>): Order =
        orders.minBy { it.createdAt }

    private fun toOrderResponse(order: Order): OrderResponse {
        val items = orderItemRepository.findByIdOrderId(order.id)
        val photoIds = items.map { it.id.photoId }
        val photos = photoRepository.findAllById(photoIds).associateBy { it.id }
        val grants = downloadGrantRepository.findByIdOrderId(order.id).associateBy { it.id.photoId }
        return OrderResponse(
            id = order.id,
            status = order.status,
            items = items.map { item ->
                val photo = photos[item.id.photoId]
                OrderResponseItem(
                    photoId = item.id.photoId,
                    price = item.pricePhpAtPurchase,
                    downloadUrl = photo?.let { downloadUrlOf(it, grants[item.id.photoId]) },
                )
            },
            totalAmount = order.totalPhp,
            paymentMethod = order.paymentMethodWire,
            createdAt = order.createdAt,
        )
    }

    private fun hydrateList(orders: List<Order>): List<OrderListItemDto> {
        if (orders.isEmpty()) return emptyList()
        val orderIds = orders.map { it.id }
        val itemsByOrder = orderItemRepository.findByIdOrderIdIn(orderIds)
            .groupBy { it.id.orderId }
        val events = eventRepository.findAllById(orders.map { it.eventId }.toSet())
            .associateBy { it.id }
        return orders.map { order ->
            val items = itemsByOrder[order.id].orEmpty()
            val event = events[order.eventId]
            OrderListItemDto(
                id = order.id,
                eventId = order.eventId,
                photoIds = items.map { it.id.photoId },
                total = order.totalPhp,
                paymentMethod = order.paymentMethodWire,
                paidAt = order.paidAt,
                eventName = event?.name,
                eventSlug = event?.slug,
                status = order.status,
            )
        }
    }

    private fun hydrateDetail(order: Order): OrderDetailDto {
        val items = orderItemRepository.findByIdOrderId(order.id)
        val photoIds = items.map { it.id.photoId }
        val photos = photoRepository.findAllById(photoIds).associateBy { it.id }
        val event = eventRepository.findById(order.eventId).orElse(null)
        val grants = downloadGrantRepository.findByIdOrderId(order.id).associateBy { it.id.photoId }
        return OrderDetailDto(
            id = order.id,
            eventId = order.eventId,
            photoIds = photoIds,
            total = order.totalPhp,
            paymentMethod = order.paymentMethodWire,
            paidAt = order.paidAt,
            eventName = event?.name,
            eventSlug = event?.slug,
            status = order.status,
            photos = items.mapIndexed { index, item ->
                val photo = photos[item.id.photoId]
                OrderPhotoDetailDto(
                    id = item.id.photoId,
                    bib = photo?.bibs?.minByOrNull { it.bibNumber }?.bibNumber,
                    time = photo?.let {
                        (it.capturedAt ?: it.uploadedAt)
                            .atZoneSameInstant(DISPLAY_ZONE)
                            .toLocalTime()
                            .format(TIME_FORMATTER)
                    } ?: "—",
                    tone = photo?.tone ?: index,
                    thumbnailUrl = photo?.let { thumbnailUrlOf(it) },
                    previewUrl = photo?.let { previewUrlOf(it) },
                    downloadUrl = photo?.let { downloadUrlOf(it, grants[item.id.photoId]) },
                )
            },
            downloadBundleUrl = null,
        )
    }

    private fun thumbnailUrlOf(photo: Photo): String {
        val key = photo.thumbnailS3Key ?: photo.watermarkS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    private fun previewUrlOf(photo: Photo): String {
        // Preview = watermarked variant if available, otherwise the thumbnail.
        val key = photo.watermarkS3Key ?: photo.thumbnailS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    private fun downloadUrlOf(photo: Photo, grant: DownloadGrant?): String? {
        if (grant == null) return null
        val now = OffsetDateTime.now()
        if (grant.grantedUntil.isBefore(now)) return null
        return storageService.presignedGetUrl(photo.s3Key, storageProperties.presignedTtl.runnerDownload)
    }

    private companion object {
        val TIME_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm")
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        // Defensive only — the controller DTO already runs Jakarta @Email via
        // BeanValidation; this regex catches the residual "looks email-shaped"
        // case if a future caller bypasses the annotated path.
        val EMAIL_REGEX: Regex = Regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")
    }
}
