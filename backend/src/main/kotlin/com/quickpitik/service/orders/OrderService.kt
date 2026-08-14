package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.PaymongoProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.orders.CreateOrderItem
import com.quickpitik.dto.orders.CreateOrderRequest
import com.quickpitik.dto.orders.OrderDetailDto
import com.quickpitik.dto.orders.OrderListItemDto
import com.quickpitik.dto.orders.OrderPhotoDetailDto
import com.quickpitik.dto.orders.OrderResponse
import com.quickpitik.dto.orders.OrderResponseItem
import com.quickpitik.dto.orders.OrderStatusDto
import com.quickpitik.dto.orders.PaymongoBilling
import com.quickpitik.dto.orders.RunnerDisputeDto
import com.quickpitik.dto.orders.PaymongoCheckoutSessionAttributes
import com.quickpitik.dto.orders.PaymongoCheckoutSessionRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionRequestEnvelope
import com.quickpitik.dto.orders.PaymongoLineItem
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
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.CartItemRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.earnings.TransactionMintingService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.security.SecureRandom
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Base64
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
    private val paymongoClient: PaymongoClient,
    private val paymongoProperties: PaymongoProperties,
    private val disputeRepository: DisputeRepository,
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val platformProperties: PlatformProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    /**
     * POST /api/v1/orders.
     *
     * Creates one Order per distinct eventId in the cart (the multi-event
     * split keeps each Order row single-event for refund attribution and to
     * match the FE's MockOrder.eventId shape). All rows share the same
     * idempotency_key, so a network retry of the same POST returns the same
     * orders + the same PayMongo Checkout URL — never a second charge.
     *
     * Flow:
     *   1. Validate items, email, payment method
     *   2. Replay-check: same idempotency_key → return existing
     *   3. Create N Orders in PENDING (one per event), one shareToken per
     *      guest order so the email link can authenticate
     *   4. Call PayMongo /v1/checkout_sessions with combined line_items
     *   5. Persist N Payment rows (status=PENDING) all carrying the cs_id
     *   6. Clear cart, return redirectUrl so FE can window.location.href to it
     *
     * Download grants + earnings rows are minted in PaymongoWebhookService
     * once PayMongo confirms payment (status flips PENDING → PAID).
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

        // Idempotency: replay returns the existing primary order + recovers the
        // PayMongo Checkout URL so the FE can resume to the same hosted page.
        val existing = orderRepository.findByIdempotencyKey(idempotencyKey)
        if (existing.isNotEmpty()) {
            val primary = pickPrimary(existing)
            val resumedUrl = resumeCheckoutUrl(primary)
            return toOrderResponse(primary, redirectUrl = resumedUrl)
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
                        status = OrderStatus.PENDING,
                        totalPhp = totalPhp,
                        idempotencyKey = idempotencyKey,
                        // Always minted — runners need it for the bundle-download
                        // URL (top-level <a> navigations don't carry the JWT, so
                        // the bundle endpoint authorizes via ?token=). Guests use
                        // it for everything (status poll, detail, bundle).
                        shareToken = generateShareToken(),
                        // V27 — the token is a bearer credential that travels
                        // in email, so it expires well before the 1-year
                        // download grant does.
                        tokenExpiresAt = OffsetDateTime.now().plus(platformProperties.shareTokenTtl),
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
                created.add(order)
            }
        } catch (ex: DataIntegrityViolationException) {
            // Concurrent retry hit the unique (idempotency_key, event_id) index.
            val replay = orderRepository.findByIdempotencyKey(idempotencyKey)
            if (replay.isNotEmpty()) {
                val primary = pickPrimary(replay)
                return toOrderResponse(primary, redirectUrl = resumeCheckoutUrl(primary))
            }
            throw ex
        }

        val primary = pickPrimary(created)
        val lineItems = buildLineItems(request.items, photos, eventLookup)
        val description = buildSessionDescription(request.items.size, eventLookup.values.map { it.name })
        // clientPlatform=="android" routes PayMongo's success/cancel to the
        // mobile bridge so the user lands back in the app via the
        // quickpitik:// deep link instead of the website.
        val isAndroid = request.clientPlatform.equals("android", ignoreCase = true)
        val successUrl = buildSuccessUrl(primary, isAndroid)
        val cancelUrlForSession = if (isAndroid) {
            buildMobileCancelUrl(primary)
        } else {
            paymongoProperties.cancelUrl
        }

        val checkout = try {
            paymongoClient.createCheckoutSession(
                PaymongoCheckoutSessionRequest(
                    data = PaymongoCheckoutSessionRequestEnvelope(
                        attributes = PaymongoCheckoutSessionAttributes(
                            cancelUrl = cancelUrlForSession,
                            successUrl = successUrl,
                            lineItems = lineItems,
                            paymentMethodTypes = paymongoMethodsFor(paymentMethod),
                            description = description,
                            billing = PaymongoBilling(email = recipientEmail),
                            metadata = mapOf(
                                "idempotencyKey" to idempotencyKey,
                                "primaryOrderId" to primary.id.toString(),
                                "orderCount" to created.size.toString(),
                            ),
                        ),
                    ),
                ),
            )
        } catch (ex: Exception) {
            log.error("PayMongo Checkout Session creation failed: {}", ex.message, ex)
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "Payment gateway unavailable — try again in a moment.",
            )
        }

        val csId = checkout.data.id
        val checkoutUrl = checkout.data.attributes.checkoutUrl
        if (csId.isBlank() || checkoutUrl.isBlank()) {
            log.error("PayMongo returned blank csId or checkoutUrl: {}", checkout)
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "Payment gateway returned an invalid response.",
            )
        }

        // N Payment rows all carrying the same cs_id. V17 UNIQUE on
        // (provider, provider_ref, order_id) keeps webhook replays clean.
        created.forEach { order ->
            paymentRepository.save(
                Payment(
                    orderId = order.id,
                    provider = "paymongo",
                    providerRef = csId,
                    amountPhp = order.totalPhp,
                    status = PaymentStatus.PENDING,
                    paidAt = null,
                ),
            )
        }

        // Cart-clear runs at order creation (PENDING) so the user doesn't see
        // duplicate items on a return-to-cart bounce. If they cancel on
        // PayMongo's page they re-add — accepted tradeoff for simpler UX.
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

        return toOrderResponse(primary, redirectUrl = checkoutUrl)
    }

    /**
     * Public guest-friendly status. Anti-IDOR: missing/mismatched token
     * surfaces the same NOT_FOUND envelope as a truly nonexistent id, so a
     * probing attacker can't enumerate order ids from the response.
     */
    @Transactional(readOnly = true)
    fun statusByIdAndToken(orderId: UUID, token: String?): OrderStatusDto {
        val order = orderRepository.findById(orderId).orElseThrow { orderNotFound() }
        requireValidToken(order, token)
        return OrderStatusDto(id = order.id, status = order.status, paidAt = order.paidAt)
    }

    /**
     * Gate for every token-authorized read. Mismatch and expiry both surface
     * NOT_FOUND rather than 401/410 — the anti-IDOR property is that probing
     * by id teaches an attacker nothing, and "this order exists but your token
     * is stale" would teach them the id was real.
     *
     * OrderBundleService.prepare applies the same rule for the streamed
     * download; it can't share this method because it resolves its own order.
     */
    private fun requireValidToken(order: Order, token: String?) {
        if (token.isNullOrBlank() || order.shareToken == null || order.shareToken != token) {
            throw orderNotFound()
        }
        if (order.tokenExpiresAt.isBefore(OffsetDateTime.now())) {
            log.info("Expired share token used for order {} (expired {})", order.id, order.tokenExpiresAt)
            throw orderNotFound()
        }
    }

    private fun orderNotFound(): NotFoundException =
        NotFoundException(code = ErrorCodes.ORDER_NOT_FOUND, message = "Order not found")

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

    private fun pickPrimary(orders: List<Order>): Order =
        orders.minBy { it.createdAt }

    private fun toOrderResponse(order: Order, redirectUrl: String? = null): OrderResponse {
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
            redirectUrl = redirectUrl,
        )
    }

    private fun generateShareToken(): String {
        val bytes = ByteArray(24)
        SecureRandom().nextBytes(bytes)
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
    }

    private fun buildLineItems(
        items: List<CreateOrderItem>,
        photos: Map<UUID, Photo>,
        events: Map<UUID, Event>,
    ): List<PaymongoLineItem> = items.map { item ->
        val photo = photos.getValue(item.photoId)
        val event = events[item.eventId]
        val bibLabel = photo.bibs.minByOrNull { it.bibNumber }?.bibNumber
            ?.let { "BIB $it" } ?: "Untagged"
        PaymongoLineItem(
            name = "Race photo · $bibLabel".take(120),
            amount = photo.pricePhp.multiply(BigDecimal(100)).toLong(),
            currency = "PHP",
            quantity = 1,
            description = event?.name?.take(120),
        )
    }

    private fun buildSessionDescription(itemCount: Int, eventNames: List<String>): String {
        val sample = eventNames.firstOrNull() ?: "QuickPitik"
        val more = if (eventNames.size > 1) " +${eventNames.size - 1} more" else ""
        val photoWord = if (itemCount == 1) "photo" else "photos"
        return "QuickPitik · $itemCount $photoWord · $sample$more".take(160)
    }

    private fun buildSuccessUrl(order: Order, isAndroid: Boolean = false): String {
        val base = if (isAndroid) paymongoProperties.mobileSuccessUrl else paymongoProperties.successUrl
        val sep = if (base.contains("?")) "&" else "?"
        // Token in the success URL is guest-only so authed runners reach
        // /orders/return via the JWT-gated /me/orders/{id} detail endpoint
        // (which keeps cross-user IDOR checks). Bundle URL embeds the token
        // regardless because the bundle endpoint is token-only.
        val tokenParam = if (order.userId == null && order.shareToken != null) {
            "&token=${order.shareToken}"
        } else {
            ""
        }
        return "$base${sep}orderId=${order.id}$tokenParam"
    }

    private fun buildMobileCancelUrl(order: Order): String {
        val base = paymongoProperties.mobileCancelUrl
        val sep = if (base.contains("?")) "&" else "?"
        return "$base${sep}orderId=${order.id}"
    }

    private fun paymongoMethodsFor(method: PaymentMethod): List<String> = when (method) {
        PaymentMethod.GCASH -> listOf("gcash")
        PaymentMethod.MAYA -> listOf("paymaya")
        PaymentMethod.CARD -> listOf("card")
    }

    // On idempotent replay, recover the PayMongo Checkout URL from the
    // existing Payment row. Already-paid orders return null (FE skips
    // redirect and shows success directly). Failures swallowed — the FE
    // will fall back to "open your orders" messaging.
    private fun resumeCheckoutUrl(order: Order): String? {
        if (order.status == OrderStatus.PAID || order.status == OrderStatus.FULFILLED) {
            return null
        }
        val payment = paymentRepository.findByOrderId(order.id)
            .firstOrNull { it.provider == "paymongo" && it.providerRef != null }
            ?: return null
        return runCatching {
            paymongoClient.retrieveCheckoutSession(payment.providerRef!!)
                .data.attributes.checkoutUrl.ifBlank { null }
        }.getOrNull()
    }

    private fun hydrateList(orders: List<Order>): List<OrderListItemDto> {
        if (orders.isEmpty()) return emptyList()
        val orderIds = orders.map { it.id }
        val itemsByOrder = orderItemRepository.findByIdOrderIdIn(orderIds)
            .groupBy { it.id.orderId }
        val events = eventRepository.findAllById(orders.map { it.eventId }.toSet())
            .associateBy { it.id }
        val disputesByOrder = hydrateDisputesByOrderId(orderIds)
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
                eventDate = event?.date,
                eventState = event?.let { EventDtoMapper.deriveAdminEventState(it) },
                status = order.status,
                disputes = disputesByOrder[order.id].orEmpty(),
            )
        }
    }

    /**
     * Batch-hydrate disputes + their admin resolution notes for a page of
     * orders. Two round-trips total regardless of N:
     *   1. SELECT * FROM disputes WHERE order_id IN (...)
     *   2. SELECT * FROM admin_decision_log
     *      WHERE target_dispute_id IN (...) ORDER BY decided_at DESC
     *
     * The decision-log query is ordered DESC and grouped by targetDisputeId
     * in memory — the first row per disputeId is the most-recent admin
     * action (resolved / denied / escalated), and its `reason` becomes the
     * runner-visible resolution note.
     */
    private fun hydrateDisputesByOrderId(orderIds: Collection<UUID>): Map<UUID, List<RunnerDisputeDto>> {
        if (orderIds.isEmpty()) return emptyMap()
        val disputes = disputeRepository.findByOrderIdIn(orderIds)
        if (disputes.isEmpty()) return emptyMap()

        val disputeIds = disputes.map { it.id }
        val latestNoteByDisputeId: Map<UUID, String?> =
            adminDecisionLogRepository.findByTargetDisputeIdInOrderByDecidedAtDesc(disputeIds)
                .groupBy { it.targetDisputeId }
                .mapNotNull { (k, v) -> if (k == null) null else k to v.first().reason }
                .toMap()

        return disputes
            .map { d ->
                d.orderId to RunnerDisputeDto(
                    id = d.id,
                    photoId = d.photoId,
                    reason = d.reasonWire,
                    note = d.note,
                    status = d.statusWire,
                    resolution = d.resolutionWire,
                    refundAmount = d.refundAmountPhp,
                    resolutionNote = latestNoteByDisputeId[d.id],
                    openedAt = d.openedAt,
                    resolvedAt = d.resolvedAt,
                    withdrawnAt = d.withdrawnAt,
                )
            }
            .groupBy({ it.first }, { it.second })
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
                    previewUrl = photo?.let { previewUrlOf(it, grants[item.id.photoId]) },
                    downloadUrl = photo?.let { downloadUrlOf(it, grants[item.id.photoId]) },
                )
            },
            downloadBundleUrl = null,
            recipientEmail = order.recipientEmail,
            shareToken = order.shareToken,
            disputes = hydrateDisputesByOrderId(listOf(order.id))[order.id].orEmpty(),
        )
    }

    /**
     * Public guest detail. Same hydration as `/me/orders/{id}` but gated on
     * the share token instead of JWT. Anti-IDOR: missing/mismatched token
     * surfaces NOT_FOUND, matching `statusByIdAndToken`.
     */
    @Transactional(readOnly = true)
    fun detailByIdAndToken(orderId: UUID, token: String?): OrderDetailDto {
        val order = orderRepository.findById(orderId).orElseThrow { orderNotFound() }
        requireValidToken(order, token)
        return hydrateDetail(order)
    }

    private fun thumbnailUrlOf(photo: Photo): String {
        val key = photo.thumbnailS3Key ?: photo.watermarkS3Key ?: photo.s3Key
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
    }

    private fun previewUrlOf(photo: Photo, grant: DownloadGrant?): String? {
        // Owned-mode preview = the clean original. OrderDetailDto is only ever
        // returned to the order's owner (JWT user_id match) or the share-token
        // holder, so by the time we render `previewUrl` the requester has
        // already proven ownership. Serving the watermarked variant here was
        // the G-2 bug — photographer mark + QuickPitik tile were baked into
        // the JPEG and still visible to paying users. Falls back to the
        // watermarked variant only when s3_key is somehow missing (defensive;
        // the column is NOT NULL).
        //
        // Gated on the same grant as downloadUrlOf: "proved ownership once" is
        // not "owns it forever". Without this the clean original stayed
        // presignable for the life of the row, outliving the entitlement that
        // paid for it — the widest leak the share token could reach.
        if (grant == null || grant.grantedUntil.isBefore(OffsetDateTime.now())) return null
        val key = photo.s3Key.ifBlank { photo.watermarkS3Key ?: photo.thumbnailS3Key.orEmpty() }
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.runnerDownload)
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
