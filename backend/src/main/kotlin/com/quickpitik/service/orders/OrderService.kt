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
import com.quickpitik.dto.orders.PaymongoCheckoutSessionAttributes
import com.quickpitik.dto.orders.PaymongoCheckoutSessionRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionRequestEnvelope
import com.quickpitik.dto.orders.PaymongoLineItem
import com.quickpitik.dto.orders.PaymongoPaymentIntentAttachAttributes
import com.quickpitik.dto.orders.PaymongoPaymentIntentAttachEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentIntentAttachRequest
import com.quickpitik.dto.orders.PaymongoPaymentIntentRequest
import com.quickpitik.dto.orders.PaymongoPaymentIntentRequestAttributes
import com.quickpitik.dto.orders.PaymongoPaymentIntentRequestEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponse
import com.quickpitik.dto.orders.PaymongoPaymentMethodRequest
import com.quickpitik.dto.orders.PaymongoPaymentMethodRequestAttributes
import com.quickpitik.dto.orders.PaymongoPaymentMethodRequestEnvelope
import com.quickpitik.dto.orders.QrPhPaymentResponse
import com.quickpitik.dto.orders.RunnerDisputeDto
import com.quickpitik.entity.DownloadGrant
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
import com.quickpitik.entity.PhotoStatus
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
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID

@Service
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
    private val paymongoClient: PaymongoClient,
    private val paymongoProperties: PaymongoProperties,
    private val disputeRepository: DisputeRepository,
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val platformProperties: PlatformProperties,
    private val orderAccessTokenService: OrderAccessTokenService,
    private val transactionTemplate: TransactionTemplate,
    private val couponService: CouponService,
    private val paymongoWebhookService: PaymongoWebhookService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    /** Reserve locally, call PayMongo without a DB transaction, then finalize locally. */
    fun create(userId: UUID?, request: CreateOrderRequest, idempotencyKey: String): OrderResponse {
        validateItems(request.items)
        val paymentMethod = PaymentMethod.fromWire(request.paymentMethod)
        val recipientEmail = resolveRecipientEmail(userId, request.recipientEmail)

        val reservation = try {
            transactionTemplate.execute {
                reserveCheckout(userId, recipientEmail, request, paymentMethod, idempotencyKey)
            } ?: error("Checkout reservation returned null")
        } catch (ex: DataIntegrityViolationException) {
            transactionTemplate.execute {
                replayAfterConstraintRace(userId, recipientEmail, request, paymentMethod, idempotencyKey)
            } ?: throw ex
        }

        val primary = pickPrimary(reservation.orders)
        if (primary.status in SETTLED_ORDER_STATUSES) {
            return toOrderResponse(primary)
        }
        if (primary.status == OrderStatus.EXPIRED) {
            throw checkoutConflict("This checkout expired. Start again with a new Idempotency-Key.")
        }

        if (paymentMethod == PaymentMethod.QRPH) {
            return createQrPhPayment(reservation, primary, recipientEmail, userId, request.items)
        }

        resumeCheckoutUrl(primary)?.let { return toOrderResponse(primary, redirectUrl = it) }

        // PayMongo forgets idempotency results after 24 hours. Once our
        // shorter checkout window has elapsed, an unlinked provider outcome
        // must be reconciled manually rather than risk creating a second,
        // independently payable session with the same key.
        val retryDeadline = minOf(
            primary.createdAt.plus(paymongoProperties.checkoutTtl),
            primary.createdAt.plusHours(PROVIDER_RETRY_WINDOW_HOURS),
        )
        if (retryDeadline.isBefore(OffsetDateTime.now())) {
            throw checkoutConflict("Payment session outcome is unknown. Contact support before retrying checkout.")
        }

        val isAndroid = request.clientPlatform.equals("android", ignoreCase = true)
        val checkout = try {
            paymongoClient.createCheckoutSession(
                request = buildCheckoutRequest(reservation, paymentMethod, recipientEmail, isAndroid),
                idempotencyKey = primary.id.toString(),
            )
        } catch (ex: Exception) {
            log.error("PayMongo Checkout Session creation failed for order {}: {}", primary.id, ex.message, ex)
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "Payment gateway unavailable - try again in a moment.",
            )
        }

        val checkoutSessionId = checkout.data.id
        val checkoutUrl = checkout.data.attributes.checkoutUrl
        if (checkoutSessionId.isBlank() || checkoutUrl.isBlank()) {
            log.error("PayMongo returned an invalid Checkout Session for order {}", primary.id)
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "Payment gateway returned an invalid response.",
            )
        }

        try {
            transactionTemplate.executeWithoutResult {
                finalizeCheckout(reservation.orders.map { it.id }, checkoutSessionId)
            }
        } catch (ex: Exception) {
            // The provider request is recoverable: metadata points back to the
            // reserved orders and the stable provider idempotency key returns
            // the same session on retry. Never release an unknown payable
            // session into a second checkout.
            log.error("PayMongo session {} could not be linked locally; retry the same checkout", checkoutSessionId, ex)
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "Payment session could not be saved - retry in a moment.",
            )
        }
        clearCartBestEffort(userId, request.items)
        return toOrderResponse(primary, redirectUrl = checkoutUrl)
    }

    private fun reserveCheckout(
        userId: UUID?,
        recipientEmail: String,
        request: CreateOrderRequest,
        paymentMethod: PaymentMethod,
        idempotencyKey: String,
    ): CheckoutReservation {
        val photos = loadAndValidatePhotos(request.items)
        val events = loadAndValidateEvents(request.items)

        findScopedOrders(userId, recipientEmail, idempotencyKey).takeIf { it.isNotEmpty() }?.let { existing ->
            val items = validateReplay(existing, request, paymentMethod)
            return CheckoutReservation(existing, photos, events, items)
        }

        val photoIds = request.items.map { it.photoId }.toSet()
        val overlapping = if (userId != null) {
            orderRepository.findOverlappingForUser(userId, photoIds, ACTIVE_ORDER_STATUSES)
        } else {
            orderRepository.findOverlappingForGuest(recipientEmail, photoIds, ACTIVE_ORDER_STATUSES)
        }
        if (overlapping.any { it.status == OrderStatus.PAID || it.status == OrderStatus.FULFILLED }) {
            throw checkoutConflict("One or more photos have already been purchased.")
        }
        if (overlapping.isNotEmpty()) {
            val keys = overlapping.mapNotNull { it.idempotencyKey }.toSet()
            if (keys.size != 1) {
                throw checkoutConflict("One or more photos already belong to another active checkout.")
            }
            val existing = findScopedOrders(userId, recipientEmail, keys.single())
            val items = validateReplay(existing, request, paymentMethod)
            return CheckoutReservation(existing, photos, events, items)
        }

        // Fresh checkout only — the replay branches above never re-resolve the
        // coupon, so a code that expired between attempts cannot strand a
        // retry whose discount is already persisted on the order rows.
        val coupon = request.couponCode?.let(couponService::reserveForCheckout)
        val discounts: Map<UUID, BigDecimal> = if (coupon == null) {
            emptyMap()
        } else {
            photos.values
                .filter { couponService.eligible(it, coupon) }
                .associate { it.id to couponService.discountFor(it, coupon) }
        }
        if (coupon != null && discounts.isEmpty()) {
            throw ValidationException(
                message = "${coupon.code} doesn't apply to any photo in this checkout",
                code = ErrorCodes.COUPON_NOT_APPLICABLE,
                field = "couponCode",
            )
        }

        val expiresAt = OffsetDateTime.now().plus(platformProperties.shareTokenTtl)
        val savedItems = mutableListOf<OrderItem>()
        val orders = request.items.groupBy { it.eventId }.map { (eventId, items) ->
            val appliedCoupon = coupon?.takeIf {
                items.any { discounts.containsKey(it.photoId) }
            }
            val total = items.fold(BigDecimal.ZERO) { sum, item ->
                sum + photos.getValue(item.photoId).pricePhp - (discounts[item.photoId] ?: BigDecimal.ZERO)
            }
            val order = orderRepository.save(
                Order(
                    userId = userId,
                    eventId = eventId,
                    recipientEmail = recipientEmail,
                    paymentMethodWire = paymentMethod.wire,
                    status = OrderStatus.PENDING,
                    totalPhp = total,
                    idempotencyKey = idempotencyKey,
                    couponCode = appliedCoupon?.code,
                    couponId = appliedCoupon?.id,
                    legacyShareTokenHash = null,
                    tokenExpiresAt = expiresAt,
                ),
            )
            items.forEach { item ->
                savedItems += orderItemRepository.save(
                    OrderItem(
                        id = OrderItemId(order.id, item.photoId),
                        pricePhpAtPurchase = photos.getValue(item.photoId).pricePhp,
                        discountPhp = discounts[item.photoId] ?: BigDecimal.ZERO,
                    ),
                )
            }
            paymentRepository.save(
                Payment(
                    orderId = order.id,
                    provider = PAYMONGO,
                    amountPhp = order.totalPhp,
                    status = PaymentStatus.PENDING,
                ),
            )
            order
        }
        return CheckoutReservation(orders, photos, events, savedItems)
    }

    private fun replayAfterConstraintRace(
        userId: UUID?,
        recipientEmail: String,
        request: CreateOrderRequest,
        paymentMethod: PaymentMethod,
        idempotencyKey: String,
    ): CheckoutReservation? {
        val existing = findScopedOrders(userId, recipientEmail, idempotencyKey)
        if (existing.isEmpty()) return null
        val items = validateReplay(existing, request, paymentMethod)
        return CheckoutReservation(
            existing,
            loadAndValidatePhotos(request.items),
            loadAndValidateEvents(request.items),
            items,
        )
    }

    // Returns the persisted items so the caller can charge exactly what was
    // reserved (price − discount) instead of re-pricing from the photo.
    private fun validateReplay(
        orders: List<Order>,
        request: CreateOrderRequest,
        paymentMethod: PaymentMethod,
    ): List<OrderItem> {
        if (orders.isEmpty()) throw checkoutConflict("Checkout is already in progress.")
        val existingItems = orderItemRepository.findByIdOrderIdIn(orders.map { it.id })
        val existingPhotoIds = existingItems.map { it.id.photoId }.toSet()
        val requestedPhotoIds = request.items.map { it.photoId }.toSet()
        // Pure string compare — never resolve the coupon on a replay.
        val requestedCoupon = request.couponCode?.let { CouponService.normalise(it) }
        val persistedCoupons = orders.mapNotNull { it.couponCode }.toSet()
        if (existingPhotoIds != requestedPhotoIds ||
            orders.any { it.paymentMethod != paymentMethod } ||
            persistedCoupons != requestedCoupon?.let(::setOf).orEmpty()
        ) {
            throw checkoutConflict("Idempotency-Key was already used for a different checkout.")
        }
        if (orders.any { it.status == OrderStatus.EXPIRED }) {
            throw checkoutConflict("This checkout expired. Start again with a new Idempotency-Key.")
        }
        return existingItems
    }

    private fun findScopedOrders(userId: UUID?, recipientEmail: String, idempotencyKey: String): List<Order> =
        if (userId != null) {
            orderRepository.findByUserIdAndIdempotencyKey(userId, idempotencyKey)
        } else {
            orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(
                recipientEmail,
                idempotencyKey,
            )
        }

    private fun buildCheckoutRequest(
        reservation: CheckoutReservation,
        paymentMethod: PaymentMethod,
        recipientEmail: String,
        isAndroid: Boolean,
    ): PaymongoCheckoutSessionRequest {
        val primary = pickPrimary(reservation.orders)
        val baseCancelUrl = if (isAndroid) buildMobileCancelUrl(primary) else paymongoProperties.cancelUrl
        return PaymongoCheckoutSessionRequest(
            data = PaymongoCheckoutSessionRequestEnvelope(
                attributes = PaymongoCheckoutSessionAttributes(
                    cancelUrl = baseCancelUrl,
                    successUrl = buildSuccessUrl(primary, isAndroid),
                    lineItems = buildLineItems(reservation.items, reservation.photos, reservation.events),
                    paymentMethodTypes = paymongoMethodsFor(paymentMethod),
                    description = buildSessionDescription(
                        reservation.items.size,
                        reservation.events.values.map { it.name },
                    ),
                    billing = PaymongoBilling(email = recipientEmail),
                    metadata = mapOf(
                        "primaryOrderId" to primary.id.toString(),
                        "orderCount" to reservation.orders.size.toString(),
                    ),
                ),
            ),
        )
    }

    private fun finalizeCheckout(
        orderIds: List<UUID>,
        checkoutSessionId: String,
        expiresAt: OffsetDateTime? = null,
    ) {
        val payments = paymentRepository.findAllByOrderIdInForUpdate(orderIds)
        if (payments.size != orderIds.size) error("Checkout payment placeholders are incomplete")
        payments.forEach { payment ->
            val existing = payment.providerRef
            if (existing != null && existing != checkoutSessionId) {
                throw checkoutConflict("Checkout is already linked to another payment session.")
            }
            payment.providerRef = checkoutSessionId
            payment.expiresAt = expiresAt
            paymentRepository.save(payment)
        }
    }

    private fun createQrPhPayment(
        reservation: CheckoutReservation,
        primary: Order,
        recipientEmail: String,
        userId: UUID?,
        items: List<CreateOrderItem>,
    ): OrderResponse {
        val existingIntentId = paymentRepository.findByOrderId(primary.id)
            .firstOrNull { it.provider == PAYMONGO && it.providerRef?.startsWith("pi_") == true }
            ?.providerRef
        val initial = try {
            existingIntentId?.let(paymongoClient::retrievePaymentIntent)
                ?: paymongoClient.createPaymentIntent(
                    PaymongoPaymentIntentRequest(
                        PaymongoPaymentIntentRequestEnvelope(
                            PaymongoPaymentIntentRequestAttributes(
                                amount = reservation.orders.fold(BigDecimal.ZERO) { sum, order ->
                                    sum + order.totalPhp
                                }.multiply(BigDecimal(100)).longValueExact(),
                                description = buildSessionDescription(
                                    reservation.items.size,
                                    reservation.events.values.map { it.name },
                                ),
                                metadata = mapOf(
                                    "primaryOrderId" to primary.id.toString(),
                                    "orderCount" to reservation.orders.size.toString(),
                                ),
                            ),
                        ),
                    ),
                    primary.id.toString(),
                )
        } catch (ex: Exception) {
            log.error("PayMongo QRPH Payment Intent creation failed for order {}: {}", primary.id, ex.message, ex)
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "Payment gateway unavailable - try again in a moment.",
            )
        }

        val intent = try {
            if (initial.data.attributes.status == "awaiting_payment_method") {
                val method = paymongoClient.createPaymentMethod(
                    PaymongoPaymentMethodRequest(
                        PaymongoPaymentMethodRequestEnvelope(
                            PaymongoPaymentMethodRequestAttributes(
                                expirySeconds = qrPhExpirySeconds(),
                                billing = PaymongoBilling(
                                    email = recipientEmail,
                                    name = recipientEmail.substringBefore('@'),
                                ),
                            ),
                        ),
                    ),
                )
                paymongoClient.attachPaymentMethod(
                    initial.data.id,
                    PaymongoPaymentIntentAttachRequest(
                        PaymongoPaymentIntentAttachEnvelope(
                            PaymongoPaymentIntentAttachAttributes(method.data.id, initial.data.attributes.clientKey),
                        ),
                    ),
                )
            } else {
                initial
            }
        } catch (ex: Exception) {
            log.error("PayMongo QRPH Payment Method attachment failed for order {}: {}", primary.id, ex.message, ex)
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "QR code could not be generated - try again in a moment.",
            )
        }

        if (intent.data.attributes.status == "succeeded") {
            linkQrPhIntent(reservation, intent)
            paymongoWebhookService.settlePaymentIntent(
                intent.data.id,
                intent.data.attributes.payments.firstOrNull()?.id,
                intent.data.attributes.metadata,
            )
            clearCartBestEffort(userId, items)
            return toOrderResponse(orderRepository.findById(primary.id).orElse(primary))
        }

        val imageUrl = intent.data.attributes.nextAction?.code?.imageUrl.orEmpty()
        if (intent.data.id.isBlank() || !imageUrl.startsWith("data:image/png;base64,")) {
            log.error(
                "PayMongo returned an invalid QRPH response for order {} status={}",
                primary.id,
                intent.data.attributes.status,
            )
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "Payment gateway returned an invalid QR code.",
            )
        }
        val updatedAt = intent.data.attributes.updatedAt.takeIf { it > 0 }
            ?.let { OffsetDateTime.ofInstant(Instant.ofEpochSecond(it), ZoneOffset.UTC) }
            ?: OffsetDateTime.now()
        val expiresAt = updatedAt.plusSeconds(qrPhExpirySeconds().toLong())
        linkQrPhIntent(reservation, intent, expiresAt)
        clearCartBestEffort(userId, items)
        return toOrderResponse(
            primary,
            qrPh = QrPhPaymentResponse(
                imageUrl = imageUrl,
                expiresAt = expiresAt,
                returnToken = primary.takeIf { it.userId == null }
                    ?.let { orderAccessTokenService.issue(it, OrderCapability.RETURN) },
            ),
        )
    }

    private fun linkQrPhIntent(
        reservation: CheckoutReservation,
        intent: PaymongoPaymentIntentResponse,
        expiresAt: OffsetDateTime? = null,
    ) {
        try {
            transactionTemplate.executeWithoutResult {
                finalizeCheckout(reservation.orders.map { it.id }, intent.data.id, expiresAt)
            }
        } catch (ex: Exception) {
            log.error("PayMongo Payment Intent {} could not be linked locally", intent.data.id, ex)
            throw ConflictException(
                code = ErrorCodes.PAYMENT_FAILED,
                message = "Payment session could not be saved - retry in a moment.",
            )
        }
    }

    private fun qrPhExpirySeconds(): Int =
        paymongoProperties.checkoutTtl.seconds.coerceIn(60, 9000).toInt()

    private fun clearCartBestEffort(userId: UUID?, items: List<CreateOrderItem>) {
        if (userId == null) return
        items.forEach { item ->
            runCatching { cartItemRepository.deleteByUserIdAndPhotoId(userId, item.photoId) }
                .onFailure { ex -> log.warn("Cart clear failed for user={} photo={}: {}", userId, item.photoId, ex.message) }
        }
    }

    @Transactional(readOnly = true)
    fun statusByIdAndToken(orderId: UUID, token: String?): OrderStatusDto {
        val order = orderRepository.findById(orderId).orElseThrow { orderNotFound() }
        requireValidReturnToken(order, token)
        return statusDto(order)
    }

    @Transactional(readOnly = true)
    fun statusForUser(userId: UUID, orderId: UUID): OrderStatusDto {
        val order = orderRepository.findById(orderId).orElseThrow { orderNotFound() }
        if (order.userId != userId) throw orderNotFound()
        return statusDto(order)
    }

    private fun statusDto(order: Order): OrderStatusDto =
        OrderStatusDto(id = order.id, status = order.status, paidAt = order.paidAt)

    private fun requireValidReturnToken(order: Order, token: String?) {
        if (!orderAccessTokenService.isValid(order, token, OrderCapability.RETURN)) throw orderNotFound()
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
        return PaginatedResponse.of(hydrateList(page.content), page.totalElements, params)
    }

    @Transactional(readOnly = true)
    fun getDetail(userId: UUID, orderId: UUID): OrderDetailDto {
        val order = orderRepository.findById(orderId).orElseThrow { orderNotFound() }
        if (order.userId != userId) throw orderNotFound()
        return hydrateDetail(order)
    }

    @Transactional(readOnly = true)
    fun detailByIdAndToken(orderId: UUID, token: String?): OrderDetailDto {
        val order = orderRepository.findById(orderId).orElseThrow { orderNotFound() }
        requireValidReturnToken(order, token)
        return hydrateDetail(order)
    }

    private fun resolveRecipientEmail(userId: UUID?, requestEmail: String?): String {
        if (userId != null) {
            return userRepository.findById(userId).orElseThrow {
                UnauthorizedException(code = ErrorCodes.UNAUTHORIZED, message = "User not found")
            }.email.trim().lowercase()
        }
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

    private fun validateItems(items: List<CreateOrderItem>) {
        if (items.isEmpty()) {
            throw ValidationException(
                message = "items must not be empty",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "items",
            )
        }
        if (items.map { it.photoId }.distinct().size != items.size) {
            throw ValidationException(
                message = "items must not contain duplicate photos",
                code = ErrorCodes.VALIDATION_ERROR,
                field = "items",
            )
        }
    }

    private fun loadAndValidatePhotos(items: List<CreateOrderItem>): Map<UUID, Photo> {
        val photoIds = items.map { it.photoId }.distinct().sorted()
        val photos = photoRepository.findAllByIdForUpdate(photoIds).associateBy { it.id }
        if (photos.size != photoIds.size) {
            throw NotFoundException(
                message = "One or more photos not found",
                code = ErrorCodes.PHOTO_NOT_FOUND,
            )
        }
        items.forEach { item ->
            val photo = photos.getValue(item.photoId)
            if (photo.eventId != item.eventId) {
                throw ValidationException(
                    message = "Photo ${item.photoId} does not belong to event ${item.eventId}",
                    code = ErrorCodes.VALIDATION_ERROR,
                    field = "items",
                )
            }
            if (photo.status != PhotoStatus.LIVE) {
                throw checkoutConflict("Only live photos can be purchased.")
            }
            if (photo.pricePhp <= BigDecimal.ZERO) {
                throw checkoutConflict("Paid checkout requires a positive photo price.")
            }
        }
        return photos
    }

    private fun loadAndValidateEvents(items: List<CreateOrderItem>): Map<UUID, Event> {
        val eventIds = items.map { it.eventId }.toSet()
        val events = eventRepository.findAllById(eventIds)
            .filter { it.deletedAt == null }
            .associateBy { it.id }
        if (events.size != eventIds.size) {
            throw NotFoundException(message = "Event not found", code = ErrorCodes.EVENT_NOT_FOUND)
        }
        events.values.firstOrNull { it.status == EventStatus.ARCHIVED }?.let { archived ->
            throw ConflictException(message = "Event ${archived.slug} is archived", code = ErrorCodes.EVENT_ARCHIVED)
        }
        return events
    }

    private fun checkoutConflict(message: String): ConflictException =
        ConflictException(code = ErrorCodes.CONFLICT, message = message)

    private fun pickPrimary(orders: List<Order>): Order = orders.minBy { it.createdAt }

    private fun toOrderResponse(
        order: Order,
        redirectUrl: String? = null,
        qrPh: QrPhPaymentResponse? = null,
    ): OrderResponse {
        val items = orderItemRepository.findByIdOrderId(order.id)
        val photos = photoRepository.findAllById(items.map { it.id.photoId }).associateBy { it.id }
        val grants = downloadGrantRepository.findByIdOrderId(order.id).associateBy { it.id.photoId }
        return OrderResponse(
            id = order.id,
            status = order.status,
            items = items.map { item ->
                OrderResponseItem(
                    photoId = item.id.photoId,
                    price = item.pricePhpAtPurchase,
                    downloadUrl = photos[item.id.photoId]?.let { downloadUrlOf(it, grants[item.id.photoId]) },
                    discount = item.discountPhp,
                )
            },
            totalAmount = order.totalPhp,
            paymentMethod = order.paymentMethodWire,
            createdAt = order.createdAt,
            redirectUrl = redirectUrl,
            couponCode = order.couponCode,
            qrPh = qrPh,
        )
    }

    // Charged from the persisted order rows, never re-priced from the photo:
    // Payment.amountPhp, the ledger and PayMongo then share one figure.
    private fun buildLineItems(
        items: List<OrderItem>,
        photos: Map<UUID, Photo>,
        events: Map<UUID, Event>,
    ): List<PaymongoLineItem> = items.map { item ->
        val photo = photos.getValue(item.id.photoId)
        val bib = photo.bibs.minByOrNull { it.bibNumber }?.bibNumber?.let { "BIB $it" } ?: "Untagged"
        PaymongoLineItem(
            name = "Race photo - $bib".take(120),
            amount = item.pricePhpAtPurchase.subtract(item.discountPhp).multiply(BigDecimal(100)).toLong(),
            description = events[photo.eventId]?.name?.take(120),
        )
    }

    private fun buildSessionDescription(itemCount: Int, eventNames: List<String>): String {
        val sample = eventNames.firstOrNull() ?: "QuickPitik"
        val more = if (eventNames.size > 1) " +${eventNames.size - 1} more" else ""
        val noun = if (itemCount == 1) "photo" else "photos"
        return "QuickPitik - $itemCount $noun - $sample$more".take(160)
    }

    private fun buildSuccessUrl(order: Order, isAndroid: Boolean): String {
        val base = if (isAndroid) paymongoProperties.mobileSuccessUrl else paymongoProperties.successUrl
        val separator = if (base.contains("?")) "&" else "?"
        val token = if (order.userId == null) {
            "&token=${orderAccessTokenService.issue(order, OrderCapability.RETURN)}"
        } else {
            ""
        }
        return "$base${separator}orderId=${order.id}$token"
    }

    private fun buildMobileCancelUrl(order: Order): String {
        val separator = if (paymongoProperties.mobileCancelUrl.contains("?")) "&" else "?"
        return "${paymongoProperties.mobileCancelUrl}${separator}orderId=${order.id}"
    }

    private fun paymongoMethodsFor(method: PaymentMethod): List<String> = when (method) {
        PaymentMethod.GCASH -> listOf("gcash")
        PaymentMethod.MAYA -> listOf("paymaya")
        PaymentMethod.CARD -> listOf("card")
        PaymentMethod.QRPH -> listOf("qrph")
    }

    private fun resumeCheckoutUrl(order: Order): String? {
        if (order.status in SETTLED_ORDER_STATUSES || order.status == OrderStatus.EXPIRED) return null
        val payment = paymentRepository.findByOrderId(order.id)
            .firstOrNull { it.provider == PAYMONGO && !it.providerRef.isNullOrBlank() }
            ?: return null
        return runCatching {
            val checkout = paymongoClient.retrieveCheckoutSession(payment.providerRef!!)
            if (checkout.data.attributes.status.equals("expired", ignoreCase = true)) null
            else checkout.data.attributes.checkoutUrl.ifBlank { null }
        }.getOrNull()
    }

    private fun hydrateList(orders: List<Order>): List<OrderListItemDto> {
        if (orders.isEmpty()) return emptyList()
        val orderIds = orders.map { it.id }
        val itemsByOrder = orderItemRepository.findByIdOrderIdIn(orderIds).groupBy { it.id.orderId }
        val events = eventRepository.findAllById(orders.map { it.eventId }.toSet()).associateBy { it.id }
        val disputes = hydrateDisputesByOrderId(orderIds)
        return orders.map { order ->
            val event = events[order.eventId]
            OrderListItemDto(
                id = order.id,
                eventId = order.eventId,
                photoIds = itemsByOrder[order.id].orEmpty().map { it.id.photoId },
                total = order.totalPhp,
                paymentMethod = order.paymentMethodWire,
                paidAt = order.paidAt,
                eventName = event?.name,
                eventSlug = event?.slug,
                eventDate = event?.date,
                eventState = event?.let { EventDtoMapper.deriveAdminEventState(it) },
                status = order.status,
                disputes = disputes[order.id].orEmpty(),
                couponCode = order.couponCode,
                discountTotal = itemsByOrder[order.id].orEmpty().sumOf { it.discountPhp },
            )
        }
    }

    private fun hydrateDisputesByOrderId(orderIds: Collection<UUID>): Map<UUID, List<RunnerDisputeDto>> {
        if (orderIds.isEmpty()) return emptyMap()
        val disputes = disputeRepository.findByOrderIdIn(orderIds)
        if (disputes.isEmpty()) return emptyMap()
        val notes = adminDecisionLogRepository
            .findByTargetDisputeIdInOrderByDecidedAtDesc(disputes.map { it.id })
            .groupBy { it.targetDisputeId }
            .mapNotNull { (id, rows) -> id?.let { it to rows.first().reason } }
            .toMap()
        return disputes.map { dispute ->
            dispute.orderId to RunnerDisputeDto(
                id = dispute.id,
                photoId = dispute.photoId,
                reason = dispute.reasonWire,
                note = dispute.note,
                status = dispute.statusWire,
                resolution = dispute.resolutionWire,
                refundAmount = dispute.refundAmountPhp,
                resolutionNote = notes[dispute.id],
                openedAt = dispute.openedAt,
                resolvedAt = dispute.resolvedAt,
                withdrawnAt = dispute.withdrawnAt,
            )
        }.groupBy({ it.first }, { it.second })
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
                        (it.capturedAt ?: it.uploadedAt).atZoneSameInstant(DISPLAY_ZONE).toLocalTime()
                            .format(TIME_FORMATTER)
                    } ?: "-",
                    tone = photo?.tone ?: index,
                    thumbnailUrl = photo?.let(::thumbnailUrlOf),
                    previewUrl = photo?.let { previewUrlOf(it, grants[item.id.photoId]) },
                    downloadUrl = photo?.let { downloadUrlOf(it, grants[item.id.photoId]) },
                )
            },
            downloadBundleUrl = null,
            recipientEmail = order.recipientEmail,
            shareToken = orderAccessTokenService.issue(order, OrderCapability.BUNDLE),
            disputes = hydrateDisputesByOrderId(listOf(order.id))[order.id].orEmpty(),
            couponCode = order.couponCode,
            discountTotal = items.sumOf { it.discountPhp },
        )
    }

    private fun thumbnailUrlOf(photo: Photo): String =
        storageService.presignedGetUrl(
            photo.thumbnailS3Key ?: photo.watermarkS3Key ?: photo.s3Key,
            storageProperties.presignedTtl.thumbnail,
        )

    private fun previewUrlOf(photo: Photo, grant: DownloadGrant?): String? {
        if (grant == null || grant.grantedUntil.isBefore(OffsetDateTime.now())) return null
        val key = photo.s3Key.ifBlank { photo.watermarkS3Key ?: photo.thumbnailS3Key.orEmpty() }
        return storageService.presignedGetUrl(key, storageProperties.presignedTtl.runnerDownload)
    }

    private fun downloadUrlOf(photo: Photo, grant: DownloadGrant?): String? {
        if (grant == null || grant.grantedUntil.isBefore(OffsetDateTime.now())) return null
        return storageService.presignedDownloadUrl(
            photo.s3Key,
            storageProperties.presignedTtl.runnerDownload,
            downloadFilenameOf(photo),
        )
    }

    private fun downloadFilenameOf(photo: Photo): String =
        com.quickpitik.service.photos.PhotoFilenames.downloadFilenameOf(photo)

    private data class CheckoutReservation(
        val orders: List<Order>,
        val photos: Map<UUID, Photo>,
        val events: Map<UUID, Event>,
        val items: List<OrderItem>,
    )

    private companion object {
        const val PAYMONGO = "paymongo"
        val ACTIVE_ORDER_STATUSES = listOf(OrderStatus.PENDING, OrderStatus.PAID, OrderStatus.FULFILLED)
        val SETTLED_ORDER_STATUSES = setOf(OrderStatus.PAID, OrderStatus.FULFILLED, OrderStatus.REFUNDED)
        val TIME_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("HH:mm")
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        val EMAIL_REGEX = Regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")
        const val PROVIDER_RETRY_WINDOW_HOURS = 23L
    }
}
