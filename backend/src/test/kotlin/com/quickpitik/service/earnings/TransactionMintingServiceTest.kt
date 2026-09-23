package com.quickpitik.service.earnings

import com.quickpitik.config.PlatformProperties
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeResolution
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
import kotlin.test.assertTrue

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
    private lateinit var transactionRepository: TransactionRepository
    private lateinit var eventPhotographerRepository: EventPhotographerRepository
    private lateinit var service: TransactionMintingService

    @BeforeEach
    fun setUp() {
        val orderRepository = Mockito.mock(OrderRepository::class.java)
        orderItemRepository = Mockito.mock(OrderItemRepository::class.java)
        val photoRepository = Mockito.mock(PhotoRepository::class.java)
        transactionRepository = Mockito.mock(TransactionRepository::class.java)
        eventPhotographerRepository = Mockito.mock(EventPhotographerRepository::class.java)
        Mockito.`when`(orderRepository.findById(order.id)).thenReturn(Optional.of(order))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(photo))
        Mockito.`when`(transactionRepository.save(anyArg())).thenAnswer { call ->
            (call.arguments[0] as Transaction).also { minted += it }
        }
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
        Mockito.verify(eventPhotographerRepository)
            .adjustSalesCounters(eventId, photographerId, 1, BigDecimal("90.00"))
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

    // Free checkout (2026-09-05): a 100% giveaway charged nothing, so it
    // earns nothing and never counts as a sale.
    @Test
    fun `a giveaway item mints no sale row`() {
        stubItems(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("150.00"), discountPhp = BigDecimal("150.00")))

        service.mintForPaidOrder(order.id)

        assertTrue(minted.isEmpty())
        Mockito.verifyNoInteractions(eventPhotographerRepository)
    }

    @Test
    fun `an existing sale row does not increment event counters again`() {
        stubItems(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("150.00")))
        stubSaleRow(kept = BigDecimal("112.50"), discount = BigDecimal.ZERO)

        service.mintForPaidOrder(order.id)

        assertTrue(minted.isEmpty())
        Mockito.verifyNoInteractions(eventPhotographerRepository)
    }

    // Refund rows are minted here too - one writer for the ledger. A full
    // refund negates the sale row outright; a partial scales both figures by
    // the fraction of the charged price returned.
    @Test
    fun `a full refund negates the sale row exactly`() {
        val original = stubSaleRow(kept = BigDecimal("90.00"), discount = BigDecimal("22.50"))
        val dispute = disputeResolvedAs(DisputeResolution.REFUND_FULL)

        service.mintRefund(dispute, BigDecimal("127.50"), photographerId)

        val refund = minted.single()
        assertEquals(BigDecimal("-90.00"), refund.amountKeptPhp)
        assertEquals(BigDecimal("-22.50"), refund.discountPhp)
        assertEquals(original.id, refund.refundOf)
        assertTrue(refund.isRefund)
        Mockito.verify(eventPhotographerRepository)
            .adjustSalesCounters(eventId, photographerId, -1, BigDecimal("-90.00"))
    }

    @Test
    fun `a partial refund scales kept and discount by the fraction returned`() {
        stubSaleRow(kept = BigDecimal("90.00"), discount = BigDecimal("22.50"))
        // Charged = 150.00 list - 22.50 coupon = 127.50; half of it comes back.
        stubItems(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("150.00"), discountPhp = BigDecimal("22.50")))
        val dispute = disputeResolvedAs(DisputeResolution.REFUND_PARTIAL)

        service.mintRefund(dispute, BigDecimal("63.75"), photographerId)

        val refund = minted.single()
        assertEquals(BigDecimal("-45.00"), refund.amountKeptPhp)
        assertEquals(BigDecimal("-11.25"), refund.discountPhp)
        Mockito.verify(eventPhotographerRepository)
            .adjustSalesCounters(eventId, photographerId, 0, BigDecimal("-45.00"))
    }

    @Test
    fun `a legacy refund without a sale row keeps the old keep-rate estimate`() {
        val dispute = disputeResolvedAs(DisputeResolution.REFUND_FULL)

        service.mintRefund(dispute, BigDecimal("94.06"), photographerId)

        val refund = minted.single()
        assertEquals(BigDecimal("-70.55"), refund.amountKeptPhp)
        assertEquals(BigDecimal.ZERO, refund.discountPhp)
        Mockito.verify(eventPhotographerRepository)
            .adjustSalesCounters(eventId, photographerId, -1, BigDecimal("-70.55"))
    }

    private fun disputeResolvedAs(resolution: DisputeResolution): Dispute =
        Dispute(orderId = order.id, photoId = photo.id, reasonWire = "other")
            .also { it.resolution = resolution }

    private fun stubSaleRow(kept: BigDecimal, discount: BigDecimal): Transaction {
        val original = Transaction(
            paidAt = order.createdAt,
            photographerId = photographerId,
            eventId = eventId,
            photoId = photo.id,
            orderId = order.id,
            amountKeptPhp = kept,
            discountPhp = discount,
        )
        Mockito.`when`(transactionRepository.findByOrderIdAndPhotoIdAndIsRefund(order.id, photo.id, false))
            .thenReturn(original)
        return original
    }

    private fun stubItems(vararg items: OrderItem) {
        Mockito.`when`(orderItemRepository.findByIdOrderId(order.id)).thenReturn(items.toList())
    }

    private fun <T> anyArg(): T = Mockito.any()
}
