package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.admin.AdminPayoutCycleDto
import com.quickpitik.dto.admin.MarkPayoutPaidRequest
import com.quickpitik.dto.orders.RefundRequest
import com.quickpitik.dto.orders.RefundResponse
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.Transaction
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminPayoutService
import com.quickpitik.service.orders.RefundService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.TestConfiguration
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Import
import org.springframework.context.annotation.Primary
import org.springframework.jdbc.core.JdbcTemplate
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import javax.sql.DataSource
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs

@Import(RefundPayoutCoverageIntegrationTest.ClockConfig::class)
class RefundPayoutCoverageIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var orderItems: OrderItemRepository
    @Autowired lateinit var disputes: DisputeRepository
    @Autowired lateinit var transactions: TransactionRepository
    @Autowired lateinit var cycles: PayoutCycleRepository
    @Autowired lateinit var decisions: AdminDecisionLogRepository
    @Autowired lateinit var refunds: RefundService
    @Autowired lateinit var payouts: AdminPayoutService
    @Autowired lateinit var clock: MutableClock
    @Autowired lateinit var dataSource: DataSource
    @Autowired lateinit var jdbc: JdbcTemplate

    @BeforeEach
    fun resetClock() {
        clock.set(NOW)
    }

    @ParameterizedTest
    @EnumSource(value = DisputeStatus::class, names = ["OPEN", "ESCALATED"])
    fun `active dispute withholds only its order-photo sale`(status: DisputeStatus) {
        val held = sale(amount = "1000", paidAt = NOW.minusDays(8))
        sale(
            photographer = held.photographer,
            photo = held.photo,
            amount = "400",
            paidAt = NOW.minusDays(8),
        )
        disputes.saveAndFlush(
            Dispute(
                orderId = held.order.id,
                photoId = held.photo.id,
                runnerId = held.runner.id,
                reasonWire = "other",
                statusWire = status.wire,
                openedAt = NOW.minusDays(2),
            ).apply {
                if (status == DisputeStatus.ESCALATED) refundStatus = "manual_review"
            },
        )

        assertMoney("400", transactions.sumCleared(held.photographer.id, NOW.minusDays(7)))
    }

    @ParameterizedTest
    @EnumSource(value = DisputeStatus::class, names = ["DENIED", "WITHDRAWN"])
    fun `terminal non-refund dispute releases the sale`(status: DisputeStatus) {
        val fixture = sale(amount = "1000", paidAt = NOW.minusDays(8))
        disputes.saveAndFlush(
            Dispute(
                orderId = fixture.order.id,
                photoId = fixture.photo.id,
                runnerId = fixture.runner.id,
                reasonWire = "other",
                statusWire = status.wire,
                openedAt = NOW.minusDays(2),
            ),
        )

        assertMoney("1000", transactions.sumCleared(fixture.photographer.id, NOW.minusDays(7)))
    }

    @Test
    fun `partial refund nets the released terminal sale exactly once`() {
        val fixture = sale(amount = "1000", paidAt = NOW.minusDays(8))
        disputes.saveAndFlush(
            Dispute(
                orderId = fixture.order.id,
                photoId = fixture.photo.id,
                runnerId = fixture.runner.id,
                reasonWire = "other",
                statusWire = DisputeStatus.RESOLVED.wire,
                refundAmountPhp = BigDecimal("250"),
                refundStatus = "succeeded",
                openedAt = NOW.minusDays(2),
            ),
        )
        transactions.saveAndFlush(
            Transaction(
                paidAt = fixture.paidAt,
                photographerId = fixture.photographer.id,
                eventId = fixture.event.id,
                photoId = fixture.photo.id,
                orderId = fixture.order.id,
                amountKeptPhp = BigDecimal("-250"),
                isRefund = true,
                refundOf = fixture.transaction.id,
            ),
        )
        assertMoney("750", transactions.sumCleared(fixture.photographer.id, NOW.minusDays(7)))
    }

    @Test
    fun `full refund nets the released terminal sale to zero`() {
        val fixture = sale(amount = "1000", paidAt = NOW.minusDays(8))
        disputes.saveAndFlush(
            Dispute(
                orderId = fixture.order.id,
                photoId = fixture.photo.id,
                runnerId = fixture.runner.id,
                reasonWire = "other",
                statusWire = DisputeStatus.RESOLVED.wire,
                refundAmountPhp = BigDecimal("1000"),
                refundStatus = "succeeded",
                openedAt = NOW.minusDays(2),
            ),
        )
        transactions.saveAndFlush(
            Transaction(
                paidAt = fixture.paidAt,
                photographerId = fixture.photographer.id,
                eventId = fixture.event.id,
                photoId = fixture.photo.id,
                orderId = fixture.order.id,
                amountKeptPhp = BigDecimal("-1000"),
                isRefund = true,
                refundOf = fixture.transaction.id,
            ),
        )

        assertMoney("0", transactions.sumCleared(fixture.photographer.id, NOW.minusDays(7)))
    }

    @Test
    fun `one photographer's dispute does not withhold another photographer's proceeds`() {
        val held = sale(amount = "1000", paidAt = NOW.minusDays(8))
        val other = sale(amount = "600", paidAt = NOW.minusDays(8))
        disputes.saveAndFlush(
            Dispute(
                orderId = held.order.id,
                photoId = held.photo.id,
                runnerId = held.runner.id,
                reasonWire = "other",
                statusWire = DisputeStatus.OPEN.wire,
                openedAt = NOW.minusDays(2),
            ),
        )

        assertMoney("0", transactions.sumCleared(held.photographer.id, NOW.minusDays(7)))
        assertMoney("600", transactions.sumCleared(other.photographer.id, NOW.minusDays(7)))
    }

    @Test
    fun `young sale stays clearing while its refund bites immediately`() {
        val fixture = sale(amount = "1000", paidAt = NOW.minusDays(1))
        transactions.saveAndFlush(
            Transaction(
                paidAt = fixture.paidAt,
                photographerId = fixture.photographer.id,
                eventId = fixture.event.id,
                photoId = fixture.photo.id,
                orderId = fixture.order.id,
                amountKeptPhp = BigDecimal("-250"),
                isRefund = true,
                refundOf = fixture.transaction.id,
            ),
        )

        assertMoney("-250", transactions.sumCleared(fixture.photographer.id, NOW.minusDays(7)))
    }

    @Test
    fun `active disputed proceeds cannot authorize manual payment`() {
        val cycle = disputedApprovedCycle()

        val error = assertFailsWith<ApiException> {
            payouts.markPaid(admin().id, cycle.id, MarkPayoutPaidRequest("manual-test"))
        }

        assertEquals(ErrorCodes.PAYOUT_NOT_COVERED, error.code)
        assertEquals(PayoutCycleStatus.APPROVED, cycles.findById(cycle.id).orElseThrow().status)
        assertFalse(decisions.findAll().any { it.targetPayoutId == cycle.id })
    }

    @Test
    fun `active disputed proceeds cannot authorize processing`() {
        val cycle = disputedApprovedCycle()

        val error = assertFailsWith<ApiException> {
            payouts.markProcessing(admin().id, cycle.id)
        }

        assertEquals(ErrorCodes.PAYOUT_NOT_COVERED, error.code)
        assertEquals(PayoutCycleStatus.APPROVED, cycles.findById(cycle.id).orElseThrow().status)
        assertFalse(decisions.findAll().any { it.targetPayoutId == cycle.id })
    }

    @Test
    fun `valid filing is visible when payout authorization reaches the clearing cutoff`() {
        val paidAt = OffsetDateTime.parse("2026-09-15T12:00:00Z")
        clock.set(OffsetDateTime.parse("2026-09-21T12:00:00Z"))
        val fixture = sale(amount = "1000", paidAt = paidAt)
        val cycle = approvedCycle(fixture)
        refunds.submit(
            fixture.runner.id,
            fixture.order.id,
            RefundRequest(listOf(fixture.photo.id), "other", "valid filing"),
        )
        clock.set(OffsetDateTime.parse("2026-09-22T12:01:00Z"))

        val error = assertFailsWith<ApiException> {
            payouts.markPaid(admin().id, cycle.id, MarkPayoutPaidRequest("manual-test"))
        }

        assertEquals(ErrorCodes.PAYOUT_NOT_COVERED, error.code)
        assertEquals(DisputeStatus.OPEN, disputes.findOpenForOrderPhoto(fixture.order.id, fixture.photo.id)?.status)
    }

    @Test
    fun `pending order with paidAt cannot persist a refund dispute`() {
        val fixture = sale(amount = "1000", paidAt = NOW.minusDays(1))
        fixture.order.status = OrderStatus.PENDING
        orders.saveAndFlush(fixture.order)

        val error = assertFailsWith<ConflictException> {
            refunds.submit(
                fixture.runner.id,
                fixture.order.id,
                RefundRequest(listOf(fixture.photo.id), "other", "unpaid"),
            )
        }

        assertEquals(ErrorCodes.CONFLICT, error.code)
        assertEquals(emptyList(), disputes.findByOrderId(fixture.order.id))
    }

    @Test
    fun `refund cutoff and payout authorization serialize on the photographer lock`() {
        val paidAt = OffsetDateTime.parse("2026-09-15T12:00:00Z")
        clock.set(OffsetDateTime.parse("2026-09-22T11:59:00Z"))
        val fixture = sale(amount = "1000", paidAt = paidAt)
        val cycle = approvedCycle(fixture)
        val adminId = admin().id
        val pool = Executors.newFixedThreadPool(2)
        val ready = CountDownLatch(2)
        val start = CountDownLatch(1)

        try {
            dataSource.connection.use { lock ->
                lock.autoCommit = false
                lock.prepareStatement("SELECT id FROM users WHERE id = ? FOR UPDATE").use {
                    it.setObject(1, fixture.photographer.id)
                    it.executeQuery().close()
                }
                val filing = pool.submit<Result<RefundResponse>> {
                    ready.countDown()
                    start.await()
                    runCatching {
                        refunds.submit(
                            fixture.runner.id,
                            fixture.order.id,
                            RefundRequest(listOf(fixture.photo.id), "other", "cutoff race"),
                        )
                    }
                }
                val payout = pool.submit<Result<AdminPayoutCycleDto>> {
                    ready.countDown()
                    start.await()
                    runCatching { payouts.markPaid(adminId, cycle.id, MarkPayoutPaidRequest("race-test")) }
                }

                assertEquals(true, ready.await(5, TimeUnit.SECONDS))
                start.countDown()
                awaitBlockedQueries(listOf(filing, payout))
                clock.set(OffsetDateTime.parse("2026-09-22T12:01:00Z"))
                lock.commit()

                assertIs<ConflictException>(filing.get(20, TimeUnit.SECONDS).exceptionOrNull())
                assertEquals("paid", payout.get(20, TimeUnit.SECONDS).getOrThrow().status)
            }
        } finally {
            pool.shutdownNow()
        }
    }

    private fun awaitBlockedQueries(futures: List<java.util.concurrent.Future<*>>) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        while (jdbc.queryForObject(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() " +
                    "AND wait_event_type = 'Lock'",
                Int::class.java,
            )!! < futures.size
        ) {
            check(futures.none { it.isDone }) { "Expected refund filing and payout authorization to wait on the user lock" }
            check(System.nanoTime() < deadline) { "Timed out waiting for refund and payout lock contention" }
            Thread.sleep(20)
        }
    }

    private fun disputedApprovedCycle(): PayoutCycle {
        val sale = sale(amount = "1000", paidAt = NOW.minusDays(8))
        disputes.saveAndFlush(
            Dispute(
                orderId = sale.order.id,
                photoId = sale.photo.id,
                runnerId = sale.runner.id,
                reasonWire = "other",
                statusWire = DisputeStatus.OPEN.wire,
                openedAt = NOW.minusDays(2),
            ),
        )
        return approvedCycle(sale)
    }

    private fun approvedCycle(sale: SaleFixture) = cycles.saveAndFlush(
        PayoutCycle(
            id = "PAY-DISPUTE-${UUID.randomUUID()}",
            photographerId = sale.photographer.id,
            weekOf = LocalDate.of(2026, 9, 14),
            amountPhp = sale.transaction.amountKeptPhp,
            statusWire = PayoutCycleStatus.APPROVED.wire,
            methodWire = "gcash",
        ),
    )

    private fun sale(
        photographer: User = photographer(),
        photo: Photo? = null,
        amount: String,
        paidAt: OffsetDateTime,
    ): SaleFixture {
        val runner = runner()
        val event = photo?.let { events.findById(it.eventId).orElseThrow() } ?: events.saveAndFlush(
            Event(
                slug = "dispute-${UUID.randomUUID()}",
                name = "Dispute payout fixture",
                date = LocalDate.of(2026, 9, 15),
                location = "Cebu",
                status = EventStatus.ACTIVE,
            ),
        )
        val ownedPhoto = photo ?: photos.saveAndFlush(
            Photo(
                eventId = event.id,
                photographerId = photographer.id,
                s3Key = "dispute/${UUID.randomUUID()}.jpg",
                pricePhp = BigDecimal(amount),
            ),
        )
        val order = orders.saveAndFlush(
            Order(
                userId = runner.id,
                eventId = event.id,
                recipientEmail = runner.email,
                paymentMethodWire = "gcash",
                status = OrderStatus.PAID,
                totalPhp = BigDecimal(amount),
                paidAt = paidAt,
            ),
        )
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, ownedPhoto.id), BigDecimal(amount)))
        val transaction = transactions.saveAndFlush(
            Transaction(
                paidAt = paidAt,
                photographerId = photographer.id,
                eventId = event.id,
                photoId = ownedPhoto.id,
                orderId = order.id,
                buyerId = runner.id,
                amountKeptPhp = BigDecimal(amount),
            ),
        )
        return SaleFixture(photographer, runner, event, ownedPhoto, order, transaction, paidAt)
    }

    private fun photographer() = user(Role.PHOTOGRAPHER)
    private fun runner() = user(Role.RUNNER)
    private fun admin() = user(Role.ADMIN)

    private fun user(role: Role) = users.saveAndFlush(
        User(
            email = "dispute-${UUID.randomUUID()}@test.local",
            passwordHash = "test-only-unused",
            name = "Dispute fixture",
            role = role,
        ),
    )

    private fun assertMoney(expected: String, actual: BigDecimal) {
        assertEquals(0, BigDecimal(expected).compareTo(actual), "expected $expected but was $actual")
    }

    private data class SaleFixture(
        val photographer: User,
        val runner: User,
        val event: Event,
        val photo: Photo,
        val order: Order,
        val transaction: Transaction,
        val paidAt: OffsetDateTime,
    )

    class MutableClock(initial: Instant) : Clock() {
        private val current = AtomicReference(initial)

        fun set(value: OffsetDateTime) {
            current.set(value.toInstant())
        }

        override fun getZone(): ZoneId = ZoneOffset.UTC
        override fun withZone(zone: ZoneId): Clock = Clock.fixed(current.get(), zone)
        override fun instant(): Instant = current.get()
    }

    @TestConfiguration
    class ClockConfig {
        @Bean
        @Primary
        fun mutableClock(): MutableClock = MutableClock(NOW.toInstant())
    }

    companion object {
        private val NOW: OffsetDateTime = OffsetDateTime.parse("2026-09-22T12:00:00Z")
    }
}
