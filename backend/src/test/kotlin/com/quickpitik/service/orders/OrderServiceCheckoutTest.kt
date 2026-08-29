package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.orders.CreateOrderItem
import com.quickpitik.dto.orders.CreateOrderRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseAttributes
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseEnvelope
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.Payment
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.exception.ConflictException
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
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

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
    private lateinit var service: OrderService
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
        val downloadGrants = Mockito.mock(DownloadGrantRepository::class.java)
        val events = Mockito.mock(EventRepository::class.java)

        Mockito.`when`(photoRepository.findAllByIdForUpdate(anyArg())).thenReturn(listOf(photo))
        Mockito.`when`(events.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(event))
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

        val platform = PlatformProperties(orderCapabilitySecret = "x".repeat(32))
        service = OrderService(
            orderRepository,
            orderItemRepository,
            paymentRepository,
            downloadGrants,
            photoRepository,
            events,
            Mockito.mock(UserRepository::class.java),
            Mockito.mock(CartItemRepository::class.java),
            Mockito.mock(StorageService::class.java),
            StorageProperties(),
            paymongoClient,
            PaymongoProperties(),
            Mockito.mock(DisputeRepository::class.java),
            Mockito.mock(AdminDecisionLogRepository::class.java),
            platform,
            OrderAccessTokenService(platform),
            testTransactionTemplate(),
        )
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
    fun `hidden photos are rejected before a payment session is created`() {
        photo.status = PhotoStatus.HIDDEN

        assertFailsWith<ConflictException> { service.create(null, request(), key) }

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

    private fun request() = CreateOrderRequest(
        items = listOf(CreateOrderItem(photo.id, eventId)),
        paymentMethod = "gcash",
        recipientEmail = email,
    )

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
