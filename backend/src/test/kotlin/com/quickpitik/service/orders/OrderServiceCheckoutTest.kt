package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PaymongoProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.PublicProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.orders.CreateOrderItem
import com.quickpitik.dto.orders.CreateOrderRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseAttributes
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseEnvelope
import com.quickpitik.dto.orders.PaymongoNextAction
import com.quickpitik.dto.orders.PaymongoPaymentIntentRequest
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponse
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponseAttributes
import com.quickpitik.dto.orders.PaymongoPaymentIntentResponseEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentAttributes
import com.quickpitik.dto.orders.PaymongoPaymentMethodResponse
import com.quickpitik.dto.orders.PaymongoPaymentMethodResponseEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentMethodRequest
import com.quickpitik.dto.orders.PaymongoPaymentResource
import com.quickpitik.dto.orders.PaymongoQrCode
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
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerCoupon
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.REFUND_SENT_TO_MANUAL_REVIEW
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.CartItemRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerCouponRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotographerAccessGuard
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.math.BigDecimal
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class OrderServiceCheckoutTest {
    @Test
    fun `checkoutLosesRaceToVerifiedRestriction`() {
        stubProvider()
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenAnswer {
            listOf(photo.apply { removalState = com.quickpitik.entity.RemovalState.REVIEW })
        }
        val failure = runCatching { service.create(null, request(), key) }.exceptionOrNull()
        assertEquals(0, newlyCreatedOrderItemsForRestrictedPhoto())
        assertTrue(failure is ConflictException)
        Mockito.verifyNoInteractions(paymongoClient)
    }

    private fun newlyCreatedOrderItemsForRestrictedPhoto() = savedItems.count { it.id.photoId == photo.id }

    private val eventId = UUID.randomUUID()
    private val photo = Photo(
        eventId = eventId,
        s3Key = "photos/test.jpg",
        pricePhp = BigDecimal("125.00"),
    )
    private val event = Event(
        id = eventId,
        slug = "cebu-marathon",
        name = "Cebu Marathon",
        date = LocalDate.of(2026, 8, 29),
        location = "Cebu City",
        status = EventStatus.ACTIVE,
    )
    private val email = "runner@example.com"
    private val key = UUID.randomUUID().toString()

    private lateinit var orderRepository: OrderRepository
    private lateinit var orderItemRepository: OrderItemRepository
    private lateinit var paymentRepository: PaymentRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var paymongoClient: PaymongoClient
    private lateinit var couponRepository: PhotographerCouponRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var userRepository: UserRepository
    private lateinit var cartItemRepository: CartItemRepository
    private lateinit var service: OrderService
    private lateinit var checkoutReconciler: PaymongoCheckoutReconciler
    private lateinit var webhookService: PaymongoWebhookService
    private lateinit var downloadGrants: DownloadGrantRepository
    private lateinit var storageService: StorageService
    private val platform = PlatformProperties(orderCapabilitySecret = "x".repeat(32))
    private var lastProviderRequest: PaymongoCheckoutSessionRequest? = null
    private val disputeRepository = Mockito.mock(DisputeRepository::class.java)
    private val decisionLogRepository = Mockito.mock(AdminDecisionLogRepository::class.java)
    private val savedOrders = mutableListOf<Order>()
    private val savedItems = mutableListOf<OrderItem>()
    private val savedPayments = mutableListOf<Payment>()

    @BeforeEach
    fun setUp() {
        orderRepository = Mockito.mock(OrderRepository::class.java)
        orderItemRepository = Mockito.mock(OrderItemRepository::class.java)
        paymentRepository = Mockito.mock(PaymentRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        Mockito.`when`(photoRepository.isDeliveryAvailable(anyArg<UUID>())).thenReturn(true)
        paymongoClient = Mockito.mock(PaymongoClient::class.java)
        couponRepository = Mockito.mock(PhotographerCouponRepository::class.java)
        downloadGrants = Mockito.mock(DownloadGrantRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        cartItemRepository = Mockito.mock(CartItemRepository::class.java)
        checkoutReconciler = Mockito.mock(PaymongoCheckoutReconciler::class.java)
        webhookService = Mockito.mock(PaymongoWebhookService::class.java)

        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo))
        Mockito.`when`(eventRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event))
        Mockito.`when`(
            orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key),
        ).thenReturn(emptyList())
        Mockito.`when`(orderRepository.findOverlappingForGuest(eqArg(email), anyArg(), anyArg()))
            .thenReturn(emptyList())
        Mockito.`when`(orderRepository.save(anyArg())).thenAnswer { call ->
            (call.arguments[0] as Order).also { order ->
                if (savedOrders.none { it.id == order.id }) savedOrders += order
            }
        }
        Mockito.`when`(orderItemRepository.save(anyArg())).thenAnswer { call ->
            (call.arguments[0] as OrderItem).also { savedItems += it }
        }
        Mockito.`when`(paymentRepository.save(anyArg())).thenAnswer { call ->
            (call.arguments[0] as Payment).also { payment ->
                if (savedPayments.none { it.id == payment.id }) savedPayments += payment
            }
        }
        Mockito.`when`(paymentRepository.findAllByOrderIdInForUpdate(anyArg())).thenAnswer { call ->
            val ids = call.getArgument<Collection<UUID>>(0)
            savedPayments.filter { it.orderId in ids }
        }
        Mockito.`when`(orderItemRepository.findByIdOrderId(anyArg())).thenAnswer { call ->
            val orderId = call.getArgument<UUID>(0)
            savedItems.filter { it.id.orderId == orderId }
        }
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo))
        Mockito.`when`(downloadGrants.findByIdOrderId(anyArg())).thenReturn(emptyList())

        service = newService()
    }

    @Test
    fun `the runner's dispute note skips the manual review hand-off row`() {
        val userId = UUID.randomUUID()
        val order = Order(
            userId = userId,
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = PaymentMethod.GCASH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal("125.00"),
        )
        val dispute = Dispute(orderId = order.id, photoId = photo.id, runnerId = userId, reasonWire = "other")
        Mockito.`when`(orderRepository.findByUserIdOrderByPaidAtDescCreatedAtDesc(anyArg(), anyArg()))
            .thenReturn(org.springframework.data.domain.PageImpl(listOf(order)))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(anyArg())).thenReturn(emptyList())
        Mockito.`when`(disputeRepository.findByOrderIdIn(anyArg())).thenReturn(listOf(dispute))
        Mockito.`when`(decisionLogRepository.findByTargetDisputeIdInOrderByDecidedAtDesc(anyArg())).thenReturn(
            listOf(
                AdminDecisionLog(adminId = UUID.randomUUID(), targetDisputeId = dispute.id, decision = REFUND_SENT_TO_MANUAL_REVIEW),
                AdminDecisionLog(adminId = UUID.randomUUID(), targetDisputeId = dispute.id, decision = "escalated", reason = "X"),
            ),
        )

        val note = service.listForUser(userId, PaginationParams(0, 20)).items.single().disputes.single().resolutionNote

        assertEquals("X", note)
    }

    @Test
    fun `the runner's dispute carries the refund status and link, or nulls when absent`() {
        val userId = UUID.randomUUID()
        val order = Order(
            userId = userId,
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = PaymentMethod.QRPH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal("125.00"),
        )
        val sentAt = OffsetDateTime.parse("2026-10-29T10:15:30Z")
        val withLink = Dispute(orderId = order.id, photoId = photo.id, runnerId = userId, reasonWire = "other")
        withLink.refundStatus = "manual_review"
        withLink.refundLinkUrl = "https://transfer.paymongo.com/abc"
        withLink.refundLinkSentAt = sentAt
        val plain = Dispute(orderId = order.id, photoId = UUID.randomUUID(), runnerId = userId, reasonWire = "other")
        Mockito.`when`(orderRepository.findByUserIdOrderByPaidAtDescCreatedAtDesc(anyArg(), anyArg()))
            .thenReturn(org.springframework.data.domain.PageImpl(listOf(order)))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(anyArg())).thenReturn(emptyList())
        Mockito.`when`(disputeRepository.findByOrderIdIn(anyArg())).thenReturn(listOf(withLink, plain))
        Mockito.`when`(decisionLogRepository.findByTargetDisputeIdInOrderByDecidedAtDesc(anyArg())).thenReturn(emptyList())

        val disputes = service.listForUser(userId, PaginationParams(0, 20)).items.single().disputes.associateBy { it.id }

        assertEquals("manual_review", disputes.getValue(withLink.id).refundStatus)
        assertEquals("https://transfer.paymongo.com/abc", disputes.getValue(withLink.id).refundLinkUrl)
        assertEquals(sentAt, disputes.getValue(withLink.id).refundLinkSentAt)
        assertEquals(sentAt.plusDays(3), disputes.getValue(withLink.id).refundLinkExpiresAt)
        assertNull(disputes.getValue(plain.id).refundStatus)
        assertNull(disputes.getValue(plain.id).refundLinkUrl)
        assertNull(disputes.getValue(plain.id).refundLinkSentAt)
        assertNull(disputes.getValue(plain.id).refundLinkExpiresAt)
    }

    @Test
    fun `the runner's link expiry is only shown while the refund waits for a manual refund`() {
        val userId = UUID.randomUUID()
        val order = Order(
            userId = userId,
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = PaymentMethod.QRPH.wire,
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal("125.00"),
        )
        val sentAt = OffsetDateTime.parse("2026-10-29T10:15:30Z")
        val done = Dispute(orderId = order.id, photoId = photo.id, runnerId = userId, reasonWire = "other")
        done.refundStatus = "succeeded"
        done.refundLinkUrl = "https://transfer.paymongo.com/abc"
        done.refundLinkSentAt = sentAt
        Mockito.`when`(orderRepository.findByUserIdOrderByPaidAtDescCreatedAtDesc(anyArg(), anyArg()))
            .thenReturn(org.springframework.data.domain.PageImpl(listOf(order)))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(anyArg())).thenReturn(emptyList())
        Mockito.`when`(disputeRepository.findByOrderIdIn(anyArg())).thenReturn(listOf(done))
        Mockito.`when`(decisionLogRepository.findByTargetDisputeIdInOrderByDecidedAtDesc(anyArg())).thenReturn(emptyList())

        val dto = service.listForUser(userId, PaginationParams(0, 20)).items.single().disputes.single()

        assertEquals(sentAt, dto.refundLinkSentAt)
        assertNull(dto.refundLinkExpiresAt)
    }

    private fun newService(
        paymongo: PaymongoProperties = PaymongoProperties(),
        platform: PlatformProperties = this.platform,
    ): OrderService = OrderService(
            orderRepository,
            orderItemRepository,
            paymentRepository,
            downloadGrants,
            photoRepository,
            eventRepository,
            userRepository,
            cartItemRepository,
            storageService,
            StorageProperties(),
            paymongoClient,
            paymongo,
            disputeRepository,
            decisionLogRepository,
            platform,
            OrderAccessTokenService(platform),
            testTransactionTemplate(),
            // Real service over a mocked repository: the discount arithmetic is
            // the thing under test, so it must not be a stub.
            CouponService(
                couponRepository,
                photoRepository,
                Mockito.mock(PhotographerSettingsRepository::class.java),
                Mockito.mock(UserRepository::class.java),
                platform,
                eventRepository,
                Mockito.mock(EventPhotographerRepository::class.java),
                orderRepository,
                PhotographerAccessGuard(
                    Mockito.mock(UserRepository::class.java),
                    Mockito.mock(PhotographerSettingsRepository::class.java),
                ),
            ),
            webhookService,
            checkoutReconciler,
            PublicProperties(apiBaseUrl = "https://api.test/api/v1"),
        )

    // 2026-09-05: the per-photo download URL is our own bundle route (+ `photo=`),
    // not a presigned R2 link — Meta's in-app browsers append `fbclid=` to the
    // navigation and R2 then rejects the SigV4 signature.
    @Test
    fun `a granted photo's download URL streams through the backend with the bundle token`() {
        val order = Order(
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = PaymentMethod.GCASH.wire,
            status = OrderStatus.PAID,
            totalPhp = BigDecimal("125.00"),
        )
        Mockito.`when`(orderRepository.findById(order.id)).thenReturn(Optional.of(order))
        savedItems += OrderItem(
            id = OrderItemId(orderId = order.id, photoId = photo.id),
            pricePhpAtPurchase = BigDecimal("125.00"),
        )
        Mockito.`when`(downloadGrants.findByIdOrderId(order.id)).thenReturn(
            listOf(
                DownloadGrant(
                    id = DownloadGrantId(orderId = order.id, photoId = photo.id),
                    grantedUntil = OffsetDateTime.now().plusYears(1),
                ),
            ),
        )
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(event))
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")
        val tokens = OrderAccessTokenService(platform)

        val detail = service.detailByIdAndToken(order.id, tokens.issue(order, OrderCapability.RETURN))

        val url = detail.photos.single().downloadUrl!!
        assertTrue(url.startsWith("https://api.test/api/v1/orders/${order.id}/download-bundle?token="), url)
        assertTrue(url.endsWith("&photo=${photo.id}"), url)
        val token = URLDecoder.decode(url.substringAfter("token=").substringBefore("&"), StandardCharsets.UTF_8)
        assertEquals(detail.shareToken, token)
        Mockito.verify(storageService, Mockito.never()).presignedDownloadUrl(anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `verify skips the provider when the order is already settled`() {
        val order = Order(
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = "qrph",
            status = com.quickpitik.entity.OrderStatus.FULFILLED,
            totalPhp = BigDecimal("125.00"),
            tokenExpiresAt = OffsetDateTime.now().plusDays(1),
        )
        Mockito.`when`(orderRepository.findById(order.id)).thenReturn(java.util.Optional.of(order))
        val token = OrderAccessTokenService(PlatformProperties(orderCapabilitySecret = "x".repeat(32)))
            .issue(order, OrderCapability.RETURN)

        val status = service.verifyByIdAndToken(order.id, token)

        assertEquals(com.quickpitik.entity.OrderStatus.FULFILLED, status.status)
        Mockito.verifyNoInteractions(checkoutReconciler)
    }

    @Test
    fun `verify asks the reconciler about a pending order and returns the fresh status`() {
        val order = Order(
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = "qrph",
            totalPhp = BigDecimal("125.00"),
            tokenExpiresAt = OffsetDateTime.now().plusDays(1),
        )
        Mockito.`when`(orderRepository.findById(order.id)).thenReturn(java.util.Optional.of(order))
        Mockito.`when`(checkoutReconciler.reconcileOrder(order.id)).thenAnswer {
            order.status = com.quickpitik.entity.OrderStatus.FULFILLED
            Unit
        }
        val token = OrderAccessTokenService(PlatformProperties(orderCapabilitySecret = "x".repeat(32)))
            .issue(order, OrderCapability.RETURN)

        val status = service.verifyByIdAndToken(order.id, token)

        assertEquals(com.quickpitik.entity.OrderStatus.FULFILLED, status.status)
        Mockito.verify(checkoutReconciler).reconcileOrder(order.id)
    }

    @Test
    fun `checkout is actor scoped and PayMongo runs outside the database transaction`() {
        var providerCallWasTransactional = true
        var providerKey = ""
        var providerRequest: PaymongoCheckoutSessionRequest? = null
        Mockito.`when`(paymongoClient.createCheckoutSession(anyArg(), anyArg())).thenAnswer { call ->
            providerCallWasTransactional = TransactionSynchronizationManager.isActualTransactionActive()
            providerRequest = call.getArgument(0)
            providerKey = call.getArgument(1)
            PaymongoCheckoutSessionResponse(
                PaymongoCheckoutSessionResponseEnvelope(
                    id = "cs_test",
                    attributes = PaymongoCheckoutSessionResponseAttributes(checkoutUrl = "https://pay.test/cs_test"),
                ),
            )
        }

        val response = service.create(null, request(), key)

        assertFalse(providerCallWasTransactional)
        assertEquals(savedOrders.single().id.toString(), providerKey)
        assertFalse(providerRequest!!.data.attributes.metadata.containsValue(key))
        assertEquals("cs_test", savedPayments.single().providerRef)
        assertEquals("https://pay.test/cs_test", response.redirectUrl)
        assertEquals(response.totalAmount, response.checkoutTotalAmount)
        Mockito.verify(orderRepository)
            .findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key)
    }

    @Test
    fun `multi-event checkout response includes the persisted provider total`() {
        val otherEventId = UUID.randomUUID()
        val otherEvent = Event(
            id = otherEventId,
            slug = "other-event",
            name = "Other Event",
            date = LocalDate.of(2026, 9, 1),
            location = "Cebu City",
            status = EventStatus.ACTIVE,
        )
        val other = Photo(eventId = otherEventId, s3Key = "photos/other.jpg", pricePhp = BigDecimal("250.00"))
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo, other))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo, other))
        Mockito.`when`(eventRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event, otherEvent))
        stubProvider()

        val response = service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(other.id, otherEventId))),
            key,
        )

        assertEquals(BigDecimal("125.00"), response.totalAmount)
        assertEquals(BigDecimal("375.00"), response.checkoutTotalAmount)
        assertEquals(37500L, lastProviderRequest!!.data.attributes.lineItems.sumOf { it.amount })
        assertEquals(BigDecimal("375.00"), savedOrders.fold(BigDecimal.ZERO) { sum, order -> sum + order.totalPhp })

        Mockito.`when`(
            orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key),
        ).thenAnswer { savedOrders.toList() }
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(anyArg())).thenAnswer { savedItems.toList() }
        Mockito.`when`(paymentRepository.findByOrderId(anyArg())).thenAnswer { call ->
            savedPayments.filter { it.orderId == call.getArgument<UUID>(0) }
        }
        Mockito.`when`(paymongoClient.retrieveCheckoutSession("cs_test")).thenReturn(
            PaymongoCheckoutSessionResponse(
                PaymongoCheckoutSessionResponseEnvelope(
                    id = "cs_test",
                    attributes = PaymongoCheckoutSessionResponseAttributes(checkoutUrl = "https://pay.test/cs_test"),
                ),
            ),
        )
        val replay = service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(other.id, otherEventId))),
            key,
        )
        assertEquals("https://pay.test/cs_test", replay.redirectUrl)
        assertEquals(BigDecimal("375.00"), replay.checkoutTotalAmount)

        savedOrders.forEach { it.status = OrderStatus.FULFILLED }
        val settled = service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(other.id, otherEventId))),
            key,
        )
        assertEquals(OrderStatus.FULFILLED, settled.status)
        assertEquals(BigDecimal("125.00"), settled.totalAmount)
        assertEquals(BigDecimal("375.00"), settled.checkoutTotalAmount)
    }

    // M-13: the bundle link (BUNDLE token, and the signed-in per-photo download
    // built from it) must not die before the one-year download grant. Binds the
    // shipped application.yml so the production default is what is tested.
    @Test
    fun `a new order's bundle link lasts as long as its one-year download grant`() {
        val environment = org.springframework.mock.env.MockEnvironment()
        org.springframework.boot.env.YamlPropertySourceLoader()
            .load("application", org.springframework.core.io.ClassPathResource("application.yml"))
            .forEach { environment.propertySources.addLast(it) }
        val shipped = org.springframework.boot.context.properties.bind.Binder.get(environment)
            .bindOrCreate("app.platform", PlatformProperties::class.java)
        Mockito.`when`(paymongoClient.createCheckoutSession(anyArg(), anyArg())).thenReturn(
            PaymongoCheckoutSessionResponse(
                PaymongoCheckoutSessionResponseEnvelope(
                    id = "cs_test",
                    attributes = PaymongoCheckoutSessionResponseAttributes(checkoutUrl = "https://pay.test/cs_test"),
                ),
            ),
        )

        newService(platform = shipped).create(null, request(), key)

        val grantEnds = OffsetDateTime.now().plusYears(1).minusMinutes(1)
        val tokenEnds = savedOrders.single().tokenExpiresAt
        assertFalse(tokenEnds.isBefore(grantEnds), "bundle link ends $tokenEnds, before the grant ($grantEnds)")
    }

    @Test
    fun `multi-event qrph checkout returns the aggregate amount and links its Payment Intent`() {
        val otherEventId = UUID.randomUUID()
        val otherEvent = Event(
            id = otherEventId,
            slug = "other-event",
            name = "Other Event",
            date = LocalDate.of(2026, 9, 1),
            location = "Cebu City",
            status = EventStatus.ACTIVE,
        )
        val other = Photo(eventId = otherEventId, s3Key = "photos/other.jpg", pricePhp = BigDecimal("250.00"))
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo, other))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo, other))
        Mockito.`when`(eventRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event, otherEvent))
        var intentRequest: PaymongoPaymentIntentRequest? = null
        var methodRequest: PaymongoPaymentMethodRequest? = null
        Mockito.`when`(paymentRepository.findByOrderId(anyArg())).thenAnswer { call ->
            savedPayments.filter { it.orderId == call.getArgument<UUID>(0) }
        }
        Mockito.`when`(paymongoClient.createPaymentIntent(anyArg(), anyArg())).thenAnswer { call ->
            intentRequest = call.getArgument(0)
            PaymongoPaymentIntentResponse(
                PaymongoPaymentIntentResponseEnvelope(
                    id = "pi_test",
                    attributes = PaymongoPaymentIntentResponseAttributes(
                        clientKey = "pi_test_client",
                        status = "awaiting_payment_method",
                    ),
                ),
            )
        }
        Mockito.`when`(paymongoClient.createPaymentMethod(anyArg())).thenAnswer { call ->
            methodRequest = call.getArgument(0)
            PaymongoPaymentMethodResponse(PaymongoPaymentMethodResponseEnvelope("pm_test"))
        }
        Mockito.`when`(paymongoClient.attachPaymentMethod(anyArg(), anyArg())).thenReturn(
            PaymongoPaymentIntentResponse(
                PaymongoPaymentIntentResponseEnvelope(
                    id = "pi_test",
                    attributes = PaymongoPaymentIntentResponseAttributes(
                        status = "awaiting_next_action",
                        nextAction = PaymongoNextAction(PaymongoQrCode("data:image/png;base64,qr")),
                        updatedAt = OffsetDateTime.now().toEpochSecond(),
                    ),
                ),
            ),
        )

        val response = service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(other.id, otherEventId)), paymentMethod = "qrph"),
            key,
        )

        assertEquals(listOf("qrph"), intentRequest!!.data.attributes.paymentMethodAllowed)
        assertEquals(37500L, intentRequest!!.data.attributes.amount)
        assertEquals(BigDecimal("125.00"), response.totalAmount)
        assertEquals(BigDecimal("375.00"), response.checkoutTotalAmount)
        assertEquals("runner", methodRequest!!.data.attributes.billing?.name)
        assertEquals("data:image/png;base64,qr", response.qrPh?.imageUrl)
        assertTrue(savedPayments.all { it.providerRef == "pi_test" })
        assertEquals(null, response.redirectUrl)
        Mockito.verify(paymongoClient, Mockito.never()).createCheckoutSession(anyArg(), anyArg())

        Mockito.`when`(
            orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key),
        ).thenAnswer { savedOrders.toList() }
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(anyArg())).thenAnswer { savedItems.toList() }
        Mockito.`when`(paymongoClient.retrievePaymentIntent("pi_test")).thenReturn(
            PaymongoPaymentIntentResponse(
                PaymongoPaymentIntentResponseEnvelope(
                    id = "pi_test",
                    attributes = PaymongoPaymentIntentResponseAttributes(
                        status = "awaiting_next_action",
                        nextAction = PaymongoNextAction(PaymongoQrCode("data:image/png;base64,qr")),
                        updatedAt = OffsetDateTime.now().toEpochSecond(),
                    ),
                ),
            ),
        )
        val replay = service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(other.id, otherEventId)), paymentMethod = "qrph"),
            key,
        )
        assertEquals(BigDecimal("125.00"), replay.totalAmount)
        assertEquals(BigDecimal("375.00"), replay.checkoutTotalAmount)
        assertEquals(response.qrPh?.imageUrl, replay.qrPh?.imageUrl)
        Mockito.verify(paymongoClient, Mockito.times(1)).createPaymentIntent(anyArg(), anyArg())

        savedOrders.clear(); savedItems.clear(); savedPayments.clear()
        Mockito.`when`(paymongoClient.createPaymentIntent(anyArg(), anyArg())).thenReturn(
            PaymongoPaymentIntentResponse(
                PaymongoPaymentIntentResponseEnvelope(
                    id = "pi_test",
                    attributes = PaymongoPaymentIntentResponseAttributes(
                        status = "succeeded",
                        payments = listOf(
                            PaymongoPaymentResource(
                                id = "pay_failed",
                                attributes = PaymongoPaymentAttributes(status = "failed"),
                            ),
                            PaymongoPaymentResource(
                                id = "pay_paid",
                                attributes = PaymongoPaymentAttributes(status = "paid"),
                            ),
                        ),
                    ),
                ),
            ),
        )
        Mockito.`when`(webhookService.settlePaymentIntent("pi_test", "pay_paid", null)).thenAnswer {
            savedOrders.forEach { it.status = OrderStatus.FULFILLED }
            emptyMap<String, Any?>()
        }
        val settled = service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(other.id, otherEventId)), paymentMethod = "qrph"),
            key,
        )
        assertEquals(OrderStatus.FULFILLED, settled.status)
        assertEquals(BigDecimal("125.00"), settled.totalAmount)
        assertEquals(BigDecimal("375.00"), settled.checkoutTotalAmount)
        assertEquals(null, settled.qrPh)
        Mockito.verify(webhookService).settlePaymentIntent("pi_test", "pay_paid", null)
    }

    @Test
    fun `qrph test-mode simulator url reaches the client only on a test key`() {
        stubQrph(testUrl = "https://test.paymongo.com/qrph/pi_test")

        assertEquals("https://test.paymongo.com/qrph/pi_test", service.create(null, request(paymentMethod = "qrph"), key).qrPh?.testUrl)

        savedOrders.clear(); savedItems.clear(); savedPayments.clear()
        service = newService(PaymongoProperties(secretKey = "sk_live_x"))
        assertEquals(null, service.create(null, request(paymentMethod = "qrph"), key).qrPh?.testUrl)
    }

    @Test
    fun `cancel asks PayMongo first and expires an order that is still pending`() {
        val userId = UUID.randomUUID()
        val order = pendingQrphOrder(userId)

        val status = service.cancelForUser(userId, order.id)

        val inOrder = Mockito.inOrder(checkoutReconciler)
        inOrder.verify(checkoutReconciler).reconcileOrder(order.id)
        inOrder.verify(checkoutReconciler).expireOrder(order.id)
        // The mocked reconciler doesn't mutate; the DTO reflects the reload.
        assertEquals(OrderStatus.PENDING, status.status)
    }

    @Test
    fun `cancel loses the race to a payment that just settled`() {
        val userId = UUID.randomUUID()
        val order = pendingQrphOrder(userId)
        Mockito.doAnswer { order.status = OrderStatus.FULFILLED; null }
            .`when`(checkoutReconciler).reconcileOrder(order.id)

        val status = service.cancelForUser(userId, order.id)

        assertEquals(OrderStatus.FULFILLED, status.status)
        Mockito.verify(checkoutReconciler, Mockito.never()).expireOrder(anyArg())
    }

    @Test
    fun `cancel on a settled order is a plain status read`() {
        val userId = UUID.randomUUID()
        val order = pendingQrphOrder(userId).also { it.status = OrderStatus.FULFILLED }

        assertEquals(OrderStatus.FULFILLED, service.cancelForUser(userId, order.id).status)
        Mockito.verifyNoInteractions(checkoutReconciler)
    }

    @Test
    fun `cancel is anti-IDOR like status`() {
        val order = pendingQrphOrder(UUID.randomUUID())

        assertFailsWith<NotFoundException> { service.cancelForUser(UUID.randomUUID(), order.id) }
        assertFailsWith<NotFoundException> { service.cancelByIdAndToken(order.id, "not-a-token") }
        Mockito.verifyNoInteractions(checkoutReconciler)
    }

    private fun pendingQrphOrder(userId: UUID): Order {
        val order = Order(
            userId = userId,
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = PaymentMethod.QRPH.wire,
            totalPhp = BigDecimal("125.00"),
        )
        Mockito.`when`(orderRepository.findById(order.id)).thenReturn(Optional.of(order))
        return order
    }

    private fun stubQrph(testUrl: String? = null) {
        Mockito.`when`(paymentRepository.findByOrderId(anyArg())).thenAnswer { call ->
            savedPayments.filter { it.orderId == call.getArgument<UUID>(0) }
        }
        Mockito.`when`(paymongoClient.createPaymentIntent(anyArg(), anyArg())).thenReturn(
            PaymongoPaymentIntentResponse(
                PaymongoPaymentIntentResponseEnvelope(
                    id = "pi_test",
                    attributes = PaymongoPaymentIntentResponseAttributes(clientKey = "pi_test_client", status = "awaiting_payment_method"),
                ),
            ),
        )
        Mockito.`when`(paymongoClient.createPaymentMethod(anyArg()))
            .thenReturn(PaymongoPaymentMethodResponse(PaymongoPaymentMethodResponseEnvelope("pm_test")))
        Mockito.`when`(paymongoClient.attachPaymentMethod(anyArg(), anyArg())).thenReturn(
            PaymongoPaymentIntentResponse(
                PaymongoPaymentIntentResponseEnvelope(
                    id = "pi_test",
                    attributes = PaymongoPaymentIntentResponseAttributes(
                        status = "awaiting_next_action",
                        nextAction = PaymongoNextAction(PaymongoQrCode("data:image/png;base64,qr", testUrl)),
                        updatedAt = OffsetDateTime.now().toEpochSecond(),
                    ),
                ),
            ),
        )
    }

    @Test
    fun `hidden photos are rejected before a payment session is created`() {
        photo.status = PhotoStatus.HIDDEN

        assertFailsWith<ConflictException> { service.create(null, request(), key) }

        Mockito.verifyNoInteractions(paymongoClient)
    }

    @Test
    fun `a client cannot claim that a photo belongs to another event`() {
        val forgedEventId = UUID.randomUUID()

        assertFailsWith<ValidationException> {
            service.create(null, request(items = listOf(CreateOrderItem(photo.id, forgedEventId))), key)
        }

        assertEquals(0, savedOrders.size)
        Mockito.verifyNoInteractions(paymongoClient)
    }

    @Test
    fun `a deleted event cannot be checked out by replaying its photo ids`() {
        event.deletedAt = OffsetDateTime.now()

        assertFailsWith<NotFoundException> {
            service.create(null, request(), key)
        }

        assertEquals(0, savedOrders.size)
        Mockito.verifyNoInteractions(paymongoClient)
    }

    @Test
    fun `draft and archived events are rejected before reserving a checkout`() {
        stubProvider()
        for (status in listOf(EventStatus.DRAFT, EventStatus.ARCHIVED)) {
            event.status = status

            val ex = assertFailsWith<ConflictException> { service.create(null, request(), key) }

            assertEquals(ErrorCodes.EVENT_ARCHIVED, ex.code)
            assertTrue(savedOrders.isEmpty())
            Mockito.verifyNoInteractions(paymentRepository, paymongoClient, cartItemRepository)
        }
    }

    @Test
    fun `stale unknown provider outcome cannot create a second checkout session`() {
        val existing = Order(
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = "gcash",
            totalPhp = photo.pricePhp,
            idempotencyKey = key,
            createdAt = OffsetDateTime.now().minusHours(1),
        )
        Mockito.`when`(
            orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key),
        ).thenReturn(listOf(existing))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(listOf(existing.id))).thenReturn(
            listOf(OrderItem(OrderItemId(existing.id, photo.id), photo.pricePhp)),
        )

        assertFailsWith<ConflictException> { service.create(null, request(), key) }

        Mockito.verifyNoInteractions(paymongoClient)
    }

    // Already-purchased net (2026-09-07): a photo the buyer already owns (a
    // PAID/FULFILLED order for the same guest email) is rejected with the
    // distinct ALREADY_PURCHASED code, so clients steer to a free re-download
    // instead of the QR-failure copy. Clients also detect this in the cart;
    // this asserts the server-side net and its distinct code.
    @Test
    fun `checkout rejects a photo the buyer already owns with the already-purchased code`() {
        Mockito.`when`(orderRepository.findOverlappingForGuest(eqArg(email), anyArg(), anyArg()))
            .thenReturn(
                listOf(
                    Order(
                        eventId = eventId,
                        recipientEmail = email,
                        paymentMethodWire = "qrph",
                        status = OrderStatus.FULFILLED,
                        totalPhp = photo.pricePhp,
                    ),
                ),
            )

        val ex = assertFailsWith<ConflictException> { service.create(null, request(), key) }

        assertEquals(ErrorCodes.ALREADY_PURCHASED, ex.code)
        assertEquals(0, savedOrders.size)
        Mockito.verifyNoInteractions(paymongoClient)
    }

    @Test
    fun `a coupon discounts only its owner's photos and every charged amount agrees`() {
        val owner = UUID.randomUUID()
        photo.photographerId = owner
        val theirs = Photo(eventId = eventId, s3Key = "photos/theirs.jpg", pricePhp = BigDecimal("150.00"))
            .also { it.photographerId = UUID.randomUUID() }
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo, theirs))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo, theirs))
        val coupon = PhotographerCoupon(
            eventId = eventId,
            photographerId = owner,
            code = "PHOTO20",
            percentOff = 20,
        )
        Mockito.`when`(couponRepository.findScopedByCodeForUpdate("PHOTO20")).thenReturn(coupon)
        stubProvider()

        val response = service.create(
            null,
            request(
                items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(theirs.id, eventId)),
                couponCode = " photo20",
            ),
            key,
        )

        // ₱125 × 0.75 × 20% = ₱18.75 off the owner's photo; the other
        // photographer's ₱150 is untouched. Recorded total, payment placeholder,
        // PayMongo line items and the response must all say the same thing.
        val order = savedOrders.single()
        assertEquals(BigDecimal("256.25"), order.totalPhp)
        assertEquals("PHOTO20", order.couponCode)
        assertEquals(coupon.id, order.couponId)
        assertEquals(BigDecimal("18.75"), savedItems.single { it.id.photoId == photo.id }.discountPhp)
        assertEquals(0, savedItems.single { it.id.photoId == theirs.id }.discountPhp.signum())
        assertEquals(BigDecimal("256.25"), savedPayments.single().amountPhp)
        val amounts = lastProviderRequest!!.data.attributes.lineItems.map { it.amount }
        assertEquals(listOf(10625L, 15000L), amounts.sorted())
        assertEquals(order.totalPhp.multiply(BigDecimal(100)).toLong(), amounts.sum())
        assertEquals("PHOTO20", response.couponCode)
        assertEquals(BigDecimal("18.75"), response.items.single { it.photoId == photo.id }.discount)
    }

    @Test
    fun `mixed cart records an event coupon only on its eligible event order`() {
        val owner = UUID.randomUUID()
        val otherEventId = UUID.randomUUID()
        val otherEvent = Event(
            id = otherEventId,
            slug = "other-event",
            name = "Other Event",
            date = LocalDate.of(2026, 9, 1),
            location = "Cebu City",
            status = EventStatus.ACTIVE,
        )
        photo.photographerId = owner
        val other = Photo(eventId = otherEventId, s3Key = "photos/other.jpg", pricePhp = BigDecimal("150.00"))
            .also { it.photographerId = owner }
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo, other))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo, other))
        Mockito.`when`(eventRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event, otherEvent))
        val coupon = PhotographerCoupon(
            eventId = eventId,
            photographerId = owner,
            code = "EVENT20",
            percentOff = 20,
        )
        Mockito.`when`(couponRepository.findScopedByCodeForUpdate("EVENT20")).thenReturn(coupon)
        stubProvider()

        service.create(
            null,
            request(
                items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(other.id, otherEventId)),
                couponCode = "EVENT20",
            ),
            key,
        )

        val discounted = savedOrders.single { it.eventId == eventId }
        val untouched = savedOrders.single { it.eventId == otherEventId }
        assertEquals("EVENT20", discounted.couponCode)
        assertEquals(coupon.id, discounted.couponId)
        assertEquals(null, untouched.couponCode)
        assertEquals(null, untouched.couponId)
        assertEquals(BigDecimal("18.75"), savedItems.single { it.id.photoId == photo.id }.discountPhp)
        assertEquals(BigDecimal.ZERO, savedItems.single { it.id.photoId == other.id }.discountPhp)
    }

    @Test
    fun `checkout rejects a coupon from another event even for the same photographer`() {
        val owner = UUID.randomUUID()
        photo.photographerId = owner
        Mockito.`when`(couponRepository.findScopedByCodeForUpdate("OTHER20")).thenReturn(
            PhotographerCoupon(
                eventId = UUID.randomUUID(),
                photographerId = owner,
                code = "OTHER20",
                percentOff = 20,
            ),
        )

        val ex = assertFailsWith<ValidationException> {
            service.create(null, request(couponCode = "OTHER20"), key)
        }

        assertEquals(ErrorCodes.COUPON_NOT_APPLICABLE, ex.code)
        assertEquals(0, savedOrders.size)
        Mockito.verifyNoInteractions(paymongoClient)
    }

    @Test
    fun `replaying an idempotency key with a different coupon code is refused`() {
        val existing = Order(
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = "gcash",
            totalPhp = photo.pricePhp,
            idempotencyKey = key,
            couponCode = "PHOTO20",
        )
        Mockito.`when`(
            orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key),
        ).thenReturn(listOf(existing))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(listOf(existing.id))).thenReturn(
            listOf(OrderItem(OrderItemId(existing.id, photo.id), photo.pricePhp)),
        )

        assertFailsWith<ConflictException> { service.create(null, request(couponCode = "OTHER5"), key) }

        Mockito.verifyNoInteractions(paymongoClient)
    }

    @Test
    fun `a same-key retry charges the persisted discount without re-validating the coupon`() {
        val existing = Order(
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = "gcash",
            totalPhp = BigDecimal("106.25"),
            idempotencyKey = key,
            couponCode = "PHOTO20",
        )
        Mockito.`when`(
            orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key),
        ).thenReturn(listOf(existing))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(listOf(existing.id))).thenReturn(
            listOf(OrderItem(OrderItemId(existing.id, photo.id), photo.pricePhp, discountPhp = BigDecimal("18.75"))),
        )
        savedPayments += Payment(orderId = existing.id, provider = "paymongo", amountPhp = existing.totalPhp)
        // The coupon has since been deleted — re-resolving it would 400 a
        // runner who is only retrying after a provider timeout.
        Mockito.`when`(couponRepository.findScopedByCodeForUpdate("PHOTO20")).thenReturn(null)
        stubProvider()

        val response = service.create(null, request(couponCode = "photo20"), key)

        assertEquals(listOf(10625L), lastProviderRequest!!.data.attributes.lineItems.map { it.amount })
        assertEquals("https://pay.test/cs_test", response.redirectUrl)
        assertEquals(BigDecimal("106.25"), response.checkoutTotalAmount)
    }

    // Auto-apply (2026-09-05). Cart: A1, A2 (photographer A, live coupon),
    // B1 (photographer B, no coupon), C1 (photographer C, live coupon), all in
    // one event. No code is typed. Each photo is priced by its own uploader's
    // coupon and nothing else; one event order carries both coupons per item.
    @Test
    fun `each photographer's live coupon applies automatically to their own photos only`() {
        val a = UUID.randomUUID()
        val b = UUID.randomUUID()
        val c = UUID.randomUUID()
        photo.photographerId = a
        val a2 = Photo(eventId = eventId, s3Key = "photos/a2.jpg", pricePhp = BigDecimal("125.00")).also { it.photographerId = a }
        val b1 = Photo(eventId = eventId, s3Key = "photos/b1.jpg", pricePhp = BigDecimal("150.00")).also { it.photographerId = b }
        val c1 = Photo(eventId = eventId, s3Key = "photos/c1.jpg", pricePhp = BigDecimal("150.00")).also { it.photographerId = c }
        val cart = listOf(photo, a2, b1, c1)
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(cart)
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(cart)
        val couponA = PhotographerCoupon(eventId = eventId, photographerId = a, code = "AAAA", percentOff = 20)
        val couponC = PhotographerCoupon(eventId = eventId, photographerId = c, code = "CCCC", percentOff = 50)
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(eventId))).thenReturn(listOf(couponA, couponC))
        stubProvider()

        val response = service.create(null, request(items = cart.map { CreateOrderItem(it.id, eventId) }), key)

        // A: 125 × 0.75 × 20% = 18.75 each. C: 150 × 0.75 × 50% = 56.25. B: none.
        val byPhoto = savedItems.associateBy { it.id.photoId }
        assertEquals(BigDecimal("18.75"), byPhoto.getValue(photo.id).discountPhp)
        assertEquals(BigDecimal("18.75"), byPhoto.getValue(a2.id).discountPhp)
        assertEquals(0, byPhoto.getValue(b1.id).discountPhp.signum())
        assertEquals(BigDecimal("56.25"), byPhoto.getValue(c1.id).discountPhp)
        assertEquals(couponA.id, byPhoto.getValue(photo.id).couponId)
        assertEquals(couponA.id, byPhoto.getValue(a2.id).couponId)
        assertEquals(null, byPhoto.getValue(b1.id).couponId)
        assertEquals(couponC.id, byPhoto.getValue(c1.id).couponId)
        val order = savedOrders.single()
        assertEquals(BigDecimal("456.25"), order.totalPhp)
        // No code was typed: the order-level snapshot stays empty.
        assertEquals(null, order.couponCode)
        assertEquals(null, order.couponId)
        assertEquals(BigDecimal("456.25"), savedPayments.single().amountPhp)
        val amounts = lastProviderRequest!!.data.attributes.lineItems.map { it.amount }
        assertEquals(order.totalPhp.multiply(BigDecimal(100)).toLong(), amounts.sum())
        assertEquals(null, response.couponCode)
    }

    @Test
    fun `a typed code and the automatic coupons apply side by side`() {
        val a = UUID.randomUUID()
        val c = UUID.randomUUID()
        photo.photographerId = a
        val c1 = Photo(eventId = eventId, s3Key = "photos/c1.jpg", pricePhp = BigDecimal("150.00")).also { it.photographerId = c }
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo, c1))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo, c1))
        val couponA = PhotographerCoupon(eventId = eventId, photographerId = a, code = "AAAA", percentOff = 20)
        val couponC = PhotographerCoupon(eventId = eventId, photographerId = c, code = "CCCC", percentOff = 50)
        Mockito.`when`(couponRepository.findScopedByCodeForUpdate("AAAA")).thenReturn(couponA)
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(eventId))).thenReturn(listOf(couponA, couponC))
        stubProvider()

        service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(c1.id, eventId)), couponCode = "aaaa"),
            key,
        )

        val order = savedOrders.single()
        assertEquals("AAAA", order.couponCode)
        assertEquals(couponA.id, order.couponId)
        assertEquals(couponA.id, savedItems.single { it.id.photoId == photo.id }.couponId)
        assertEquals(couponC.id, savedItems.single { it.id.photoId == c1.id }.couponId)
        assertEquals(BigDecimal("200.00"), order.totalPhp)
    }

    // Free checkout (2026-09-05): a 100% giveaway on the photographer's own
    // event leaves nothing to collect. The reservation settles itself through
    // the webhook service and the answer is already FULFILLED — no provider
    // round-trip — and a guest still gets a receipt link.
    @Test
    fun `a giveaway checkout settles without the provider and answers fulfilled`() {
        val owner = UUID.randomUUID()
        photo.photographerId = owner
        val giveaway = PhotographerCoupon(eventId = eventId, photographerId = owner, code = "FREE100", percentOff = 100)
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(eventId))).thenReturn(listOf(giveaway))
        Mockito.doAnswer {
            savedOrders.forEach { it.status = OrderStatus.FULFILLED }
            null
        }.`when`(webhookService).settleFree(anyArg())

        val response = service.create(null, request(paymentMethod = "qrph"), key)

        val order = savedOrders.single()
        assertEquals(0, order.totalPhp.signum())
        assertEquals(BigDecimal("125.00"), savedItems.single().discountPhp)
        assertEquals(giveaway.id, savedItems.single().couponId)
        assertEquals(0, savedPayments.single().amountPhp.signum())
        Mockito.verify(webhookService).settleFree(listOf(order.id))
        assertEquals(OrderStatus.FULFILLED, response.status)
        assertEquals(0, response.totalAmount.signum())
        assertEquals(0, response.checkoutTotalAmount.signum())
        assertEquals(null, response.qrPh)
        assertFalse(response.returnToken.isNullOrBlank())
        Mockito.verifyNoInteractions(paymongoClient)
    }

    @Test
    fun `a signed-in giveaway checkout clears its cart item after settlement`() {
        val user = User(email = email, passwordHash = "unused", name = "Runner", role = Role.RUNNER)
        Mockito.`when`(userRepository.findById(user.id)).thenReturn(Optional.of(user))
        val owner = UUID.randomUUID()
        photo.photographerId = owner
        val giveaway = PhotographerCoupon(eventId = eventId, photographerId = owner, code = "FREE100", percentOff = 100)
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(eventId))).thenReturn(listOf(giveaway))
        Mockito.doAnswer {
            savedOrders.forEach { it.status = OrderStatus.FULFILLED }
            null
        }.`when`(webhookService).settleFree(anyArg())
        var statusAtCartClear: OrderStatus? = null
        var cartClearWasTransactional: Boolean? = null
        Mockito.`when`(cartItemRepository.deleteByUserIdAndPhotoId(user.id, photo.id)).thenAnswer {
            statusAtCartClear = savedOrders.single().status
            cartClearWasTransactional = TransactionSynchronizationManager.isActualTransactionActive()
            1
        }

        val response = service.create(user.id, request(paymentMethod = "qrph"), key)

        assertEquals(OrderStatus.FULFILLED, response.status)
        assertEquals(OrderStatus.FULFILLED, statusAtCartClear)
        assertEquals(false, cartClearWasTransactional)
        Mockito.verify(cartItemRepository).deleteByUserIdAndPhotoId(user.id, photo.id)
        Mockito.verifyNoMoreInteractions(cartItemRepository)
        Mockito.verifyNoInteractions(paymongoClient)
    }

    @Test
    fun `a giveaway beside a paid photo still charges the paid photo`() {
        val owner = UUID.randomUUID()
        photo.photographerId = owner
        val theirs = Photo(eventId = eventId, s3Key = "photos/theirs.jpg", pricePhp = BigDecimal("150.00"))
            .also { it.photographerId = UUID.randomUUID() }
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo, theirs))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo, theirs))
        val giveaway = PhotographerCoupon(eventId = eventId, photographerId = owner, code = "FREE100", percentOff = 100)
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(eventId))).thenReturn(listOf(giveaway))
        stubProvider()

        service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(theirs.id, eventId))),
            key,
        )

        assertEquals(BigDecimal("150.00"), savedOrders.single().totalPhp)
        Mockito.verify(webhookService, Mockito.never()).settleFree(anyArg())
        // The free photo sends no zero-amount line to the provider.
        assertEquals(listOf(15000L), lastProviderRequest!!.data.attributes.lineItems.map { it.amount })
    }

    @Test
    fun `a giveaway order beside a paid event order waits for the group's payment`() {
        val owner = UUID.randomUUID()
        val otherEventId = UUID.randomUUID()
        val otherEvent = Event(
            id = otherEventId,
            slug = "other-event",
            name = "Other Event",
            date = LocalDate.of(2026, 9, 1),
            location = "Cebu City",
            status = EventStatus.ACTIVE,
        )
        photo.photographerId = owner
        val other = Photo(eventId = otherEventId, s3Key = "photos/other.jpg", pricePhp = BigDecimal("150.00"))
            .also { it.photographerId = owner }
        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo, other))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo, other))
        Mockito.`when`(eventRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event, otherEvent))
        val giveaway = PhotographerCoupon(eventId = eventId, photographerId = owner, code = "FREE100", percentOff = 100)
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(eventId, otherEventId)))
            .thenReturn(listOf(giveaway))
        stubProvider()

        val response = service.create(
            null,
            request(items = listOf(CreateOrderItem(photo.id, eventId), CreateOrderItem(other.id, otherEventId))),
            key,
        )

        assertEquals(0, response.totalAmount.signum())
        assertEquals(BigDecimal("150.00"), response.checkoutTotalAmount)
        assertEquals(setOf(0, 1), savedOrders.map { it.totalPhp.signum() }.toSet())
        assertTrue(savedOrders.all { it.status == OrderStatus.PENDING })
        Mockito.verify(webhookService, Mockito.never()).settleFree(anyArg())
        assertEquals(listOf(15000L), lastProviderRequest!!.data.attributes.lineItems.map { it.amount })
    }

    @Test
    fun `replaying a settled giveaway key answers the same fulfilled order`() {
        val existing = Order(
            eventId = eventId,
            recipientEmail = email,
            paymentMethodWire = "qrph",
            status = OrderStatus.FULFILLED,
            totalPhp = BigDecimal.ZERO,
            idempotencyKey = key,
        )
        Mockito.`when`(
            orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key),
        ).thenReturn(listOf(existing))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(listOf(existing.id))).thenReturn(
            listOf(OrderItem(OrderItemId(existing.id, photo.id), photo.pricePhp, discountPhp = photo.pricePhp)),
        )

        val response = service.create(null, request(paymentMethod = "qrph"), key)

        assertEquals(existing.id, response.id)
        assertEquals(OrderStatus.FULFILLED, response.status)
        assertEquals(0, response.checkoutTotalAmount.signum())
        assertFalse(response.returnToken.isNullOrBlank())
        Mockito.verify(webhookService, Mockito.never()).settleFree(anyArg())
        Mockito.verifyNoInteractions(paymongoClient)
    }

    // Security audit 2026-10-14, orders/guest-checkout-resumed-by-unverified-email.
    @Test
    fun `a guest request under a new key is never handed another key's pending order`() {
        val plantedKey = UUID.randomUUID().toString()
        val planted = Order(
            userId = null, eventId = eventId, recipientEmail = email,
            paymentMethodWire = PaymentMethod.GCASH.wire, status = OrderStatus.PENDING,
            totalPhp = photo.pricePhp, idempotencyKey = plantedKey,
        )
        Mockito.`when`(orderRepository.findOverlappingForGuest(eqArg(email), anyArg(), anyArg()))
            .thenReturn(listOf(planted))
        Mockito.`when`(orderRepository.findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, plantedKey))
            .thenReturn(listOf(planted))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(anyArg())).thenReturn(listOf(
            OrderItem(id = OrderItemId(planted.id, photo.id), pricePhpAtPurchase = photo.pricePhp, discountPhp = BigDecimal.ZERO),
        ))
        val plantedPayment = Payment(orderId = planted.id, provider = "paymongo", amountPhp = planted.totalPhp,
            status = PaymentStatus.PENDING).apply { providerRef = "cs_planted" }
        Mockito.`when`(paymentRepository.findByOrderId(planted.id)).thenReturn(listOf(plantedPayment))
        Mockito.`when`(paymongoClient.retrieveCheckoutSession("cs_planted")).thenReturn(
            PaymongoCheckoutSessionResponse(PaymongoCheckoutSessionResponseEnvelope(
                id = "cs_planted",
                attributes = PaymongoCheckoutSessionResponseAttributes(checkoutUrl = "https://pay.test/planted"),
            )),
        )

        val ex = assertFailsWith<ConflictException> { service.create(null, request(), key) }
        assertEquals(ErrorCodes.CHECKOUT_IN_PROGRESS, ex.code)
        // An unlinked pending order never expires, so the way out is signing in.
        val message = ex.message.orEmpty()
        assertTrue("sign in" in message && "expire" !in message, message)
        Mockito.verify(paymongoClient, Mockito.never()).retrieveCheckoutSession(anyArg())
    }

    @Test
    fun `a signed-in buyer's retry under a new key resumes their own pending order`() {
        val userId = UUID.randomUUID()
        val firstKey = UUID.randomUUID().toString()
        val own = Order(userId = userId, eventId = eventId, recipientEmail = email,
            paymentMethodWire = PaymentMethod.GCASH.wire, status = OrderStatus.PENDING,
            totalPhp = photo.pricePhp, idempotencyKey = firstKey)
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(
            User(id = userId, email = email, passwordHash = "unused", name = "Runner", role = Role.RUNNER),
        ))
        Mockito.`when`(orderRepository.findByUserIdAndIdempotencyKey(userId, key)).thenReturn(emptyList())
        Mockito.`when`(orderRepository.findOverlappingForUser(eqArg(userId), anyArg(), anyArg())).thenReturn(listOf(own))
        Mockito.`when`(orderRepository.findByUserIdAndIdempotencyKey(userId, firstKey)).thenReturn(listOf(own))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(anyArg())).thenReturn(listOf(
            OrderItem(id = OrderItemId(own.id, photo.id), pricePhpAtPurchase = photo.pricePhp, discountPhp = BigDecimal.ZERO),
        ))
        savedPayments += Payment(orderId = own.id, provider = "paymongo", amountPhp = own.totalPhp)
        stubProvider()

        val response = service.create(userId, request(), key)

        assertEquals(own.id, response.id)
    }

    private fun stubProvider() {
        Mockito.`when`(paymongoClient.createCheckoutSession(anyArg(), anyArg())).thenAnswer { call ->
            lastProviderRequest = call.getArgument(0)
            PaymongoCheckoutSessionResponse(
                PaymongoCheckoutSessionResponseEnvelope(
                    id = "cs_test",
                    attributes = PaymongoCheckoutSessionResponseAttributes(checkoutUrl = "https://pay.test/cs_test"),
                ),
            )
        }
    }

    private fun request(
        items: List<CreateOrderItem> = listOf(CreateOrderItem(photo.id, eventId)),
        couponCode: String? = null,
        paymentMethod: String = "gcash",
    ) = CreateOrderRequest(
        items = items,
        paymentMethod = paymentMethod,
        recipientEmail = email,
        couponCode = couponCode,
    )

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
