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
import com.quickpitik.dto.orders.PaymongoPaymentMethodResponse
import com.quickpitik.dto.orders.PaymongoPaymentMethodResponseEnvelope
import com.quickpitik.dto.orders.PaymongoPaymentMethodRequest
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
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerCoupon
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
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
import kotlin.test.assertTrue

class OrderServiceCheckoutTest {
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
    private lateinit var service: OrderService
    private lateinit var checkoutReconciler: PaymongoCheckoutReconciler
    private lateinit var downloadGrants: DownloadGrantRepository
    private lateinit var storageService: StorageService
    private val platform = PlatformProperties(orderCapabilitySecret = "x".repeat(32))
    private var lastProviderRequest: PaymongoCheckoutSessionRequest? = null
    private val savedOrders = mutableListOf<Order>()
    private val savedItems = mutableListOf<OrderItem>()
    private val savedPayments = mutableListOf<Payment>()

    @BeforeEach
    fun setUp() {
        orderRepository = Mockito.mock(OrderRepository::class.java)
        orderItemRepository = Mockito.mock(OrderItemRepository::class.java)
        paymentRepository = Mockito.mock(PaymentRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        paymongoClient = Mockito.mock(PaymongoClient::class.java)
        couponRepository = Mockito.mock(PhotographerCouponRepository::class.java)
        downloadGrants = Mockito.mock(DownloadGrantRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        checkoutReconciler = Mockito.mock(PaymongoCheckoutReconciler::class.java)

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

        service = OrderService(
            orderRepository,
            orderItemRepository,
            paymentRepository,
            downloadGrants,
            photoRepository,
            eventRepository,
            Mockito.mock(UserRepository::class.java),
            Mockito.mock(CartItemRepository::class.java),
            storageService,
            StorageProperties(),
            paymongoClient,
            PaymongoProperties(),
            Mockito.mock(DisputeRepository::class.java),
            Mockito.mock(AdminDecisionLogRepository::class.java),
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
            ),
            Mockito.mock(PaymongoWebhookService::class.java),
            checkoutReconciler,
            PublicProperties(apiBaseUrl = "https://api.test/api/v1"),
        )
    }

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
        Mockito.verify(orderRepository)
            .findByUserIdIsNullAndRecipientEmailIgnoreCaseAndIdempotencyKey(email, key)
    }

    @Test
    fun `qrph checkout returns the generated QR and links its Payment Intent`() {
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

        val response = service.create(null, request(paymentMethod = "qrph"), key)

        assertEquals(listOf("qrph"), intentRequest!!.data.attributes.paymentMethodAllowed)
        assertEquals(12500L, intentRequest!!.data.attributes.amount)
        assertEquals("runner", methodRequest!!.data.attributes.billing?.name)
        assertEquals("data:image/png;base64,qr", response.qrPh?.imageUrl)
        assertEquals("pi_test", savedPayments.single().providerRef)
        assertEquals(null, response.redirectUrl)
        Mockito.verify(paymongoClient, Mockito.never()).createCheckoutSession(anyArg(), anyArg())
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
