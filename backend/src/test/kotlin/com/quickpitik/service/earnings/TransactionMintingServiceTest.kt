package com.quickpitik.service.earnings

import com.quickpitik.config.PlatformProperties
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Transaction
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

// The single commission multiply in the codebase. A coupon lowers what the
// photographer keeps by exactly the discount; the platform's share of the
// list price is untouched.
class TransactionMintingServiceTest {
    private val eventId = UUID.randomUUID()
    private val photographerId = UUID.randomUUID()
    private val order = Order(
        eventId = eventId,
        recipientEmail = "runner@example.com",
        paymentMethodWire = "gcash",
        totalPhp = BigDecimal("127.50"),
    )
    private val photo = Photo(eventId = eventId, s3Key = "photos/p.jpg", pricePhp = BigDecimal("150.00"))
        .also { it.photographerId = photographerId }
    private val minted = mutableListOf<Transaction>()

    private lateinit var orderItemRepository: OrderItemRepository
    private lateinit var service: TransactionMintingService

    @BeforeEach
    fun setUp() {
        val orderRepository = Mockito.mock(OrderRepository::class.java)
        orderItemRepository = Mockito.mock(OrderItemRepository::class.java)
        val photoRepository = Mockito.mock(PhotoRepository::class.java)
        val transactionRepository = Mockito.mock(TransactionRepository::class.java)
        val eventPhotographerRepository = Mockito.mock(EventPhotographerRepository::class.java)
        Mockito.`when`(orderRepository.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo))
        Mockito.`when`(transactionRepository.save(anyArg())).thenAnswer { call ->
            (call.arguments[0] as Transaction).also { minted += it }
        }
        Mockito.`when`(eventPhotographerRepository.findById(anyArg())).thenReturn(Optional.empty())
        Mockito.`when`(eventPhotographerRepository.save(anyArg())).thenAnswer { it.arguments[0] }
        service = TransactionMintingService(
            orderRepository,
            orderItemRepository,
            photoRepository,
            Mockito.mock(UserRepository::class.java),
            transactionRepository,
            eventPhotographerRepository,
            PlatformProperties(),
        )
    }

    @Test
    fun `a coupon item keeps the photographer share minus the discount and records the discount`() {
        stubItems(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("150.00"), discountPhp = BigDecimal("22.50")))

        service.mintForPaidOrder(order.id)

        val row = minted.single()
        assertEquals(BigDecimal("90.00"), row.amountKeptPhp)
        assertEquals(BigDecimal("22.50"), row.discountPhp)
        // Runner paid 127.50; platform fee = paid − kept = 37.50 = 25% of the list price.
        assertEquals(BigDecimal("37.50"), BigDecimal("127.50").subtract(row.amountKeptPhp))
    }

    @Test
    fun `an item without a coupon is unchanged`() {
        stubItems(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("150.00")))

        service.mintForPaidOrder(order.id)

        val row = minted.single()
        assertEquals(BigDecimal("112.50"), row.amountKeptPhp)
        assertEquals(0, row.discountPhp.signum())
    }

    private fun stubItems(vararg items: OrderItem) {
        Mockito.`when`(orderItemRepository.findByIdOrderId(order.id)).thenReturn(items.toList())
    }

    private fun <T> anyArg(): T = Mockito.any()
}
