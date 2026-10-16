package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.admin.AdminPayoutService
import com.quickpitik.service.earnings.PayoutRequestService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNull

// Security audit 2026-10-14, earnings/payout-cycle-id-collision-overwrites-other-cycle.
// Weeks in 2001 keep the weekly generator away from other test classes' sales.
class PayoutCycleIdCollisionIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var settings: PhotographerSettingsRepository
    @Autowired lateinit var cycles: PayoutCycleRepository
    @Autowired lateinit var accounts: PayoutAccountRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var transactions: TransactionRepository
    @Autowired lateinit var requests: PayoutRequestService
    @Autowired lateinit var admin: AdminPayoutService

    private val tag = UUID.randomUUID().toString().replace("-", "").take(8)
    private val manila = ZoneId.of("Asia/Manila")

    private fun photographer(handle: String?): User {
        val user = users.saveAndFlush(User(
            email = "collide-${UUID.randomUUID()}@test.local", passwordHash = "test-only-unused",
            name = "Collision fixture", role = Role.PHOTOGRAPHER,
        ))
        settings.saveAndFlush(PhotographerSettings(userId = user.id, handle = handle))
        return user
    }

    private fun sale(photographerId: UUID, amount: String, paidAt: OffsetDateTime) {
        val event = events.saveAndFlush(Event(slug = "collide-${UUID.randomUUID()}", name = "Collision",
            date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        val photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = photographerId,
            s3Key = "collide/${UUID.randomUUID()}", pricePhp = BigDecimal(amount)))
        val order = orders.saveAndFlush(Order(eventId = event.id, recipientEmail = "collide@test.local",
            paymentMethodWire = "gcash", status = OrderStatus.PAID, totalPhp = BigDecimal(amount)))
        transactions.saveAndFlush(Transaction(paidAt = paidAt, photographerId = photographerId,
            eventId = event.id, photoId = photo.id, orderId = order.id, amountKeptPhp = BigDecimal(amount)))
    }

    private fun primaryAccount(userId: UUID, name: String) = accounts.saveAndFlush(PayoutAccount(
        userId = userId, method = PayoutMethod.GCASH, accountNumber = "09171234567",
        accountName = name, isPrimary = true, qrS3Key = "collide/qr-$userId.png",
    ))

    @Test
    fun `a payout request never writes onto another photographer's cycle`() {
        val a = photographer("jd$tag")    // cleans to JD<TAG>
        val b = photographer("j-d$tag")   // cleans to the same letters
        val victimId = "PAY-JD${tag.uppercase()}-001"
        cycles.saveAndFlush(PayoutCycle(id = victimId, photographerId = a.id, weekOf = LocalDate.now(),
            amountPhp = BigDecimal("5000.00"), statusWire = PayoutCycleStatus.PAID.wire))
        sale(b.id, "1000.00", OffsetDateTime.now().minusDays(10))
        primaryAccount(b.id, "B account")

        runCatching { requests.request(b.id) }

        val victim = cycles.findById(victimId).get()
        assertEquals(a.id, victim.photographerId)
        assertEquals(PayoutCycleStatus.PAID, victim.status)
        assertEquals(0, BigDecimal("5000.00").compareTo(victim.amountPhp))
        assertNull(victim.accountSnapshotName)
        val own = cycles.findAll().filter { it.photographerId == b.id }
        assertEquals(1, own.size)
        assertNotEquals(victimId, own.single().id)
    }

    @Test
    fun `the weekly generator gives two photographers without a handle two cycles`() {
        val monday = LocalDate.of(2001, 1, 1)  // a Monday
        val p1 = photographer(null)
        val p2 = photographer(null)
        sale(p1.id, "700.00", monday.plusDays(1).atStartOfDay(manila).toOffsetDateTime())
        sale(p2.id, "900.00", monday.plusDays(2).atStartOfDay(manila).toOffsetDateTime())

        admin.generateForWeek(monday)

        val c1 = cycles.findAll().filter { it.photographerId == p1.id }
        val c2 = cycles.findAll().filter { it.photographerId == p2.id }
        assertEquals(1, c1.size)
        assertEquals(1, c2.size)
        assertEquals(0, BigDecimal("700.00").compareTo(c1.single().amountPhp))
        assertEquals(0, BigDecimal("900.00").compareTo(c2.single().amountPhp))
    }

    @Test
    fun `the weekly generator leaves another photographer's legacy row alone`() {
        val monday = LocalDate.of(2001, 1, 8)
        val a = photographer("wk$tag")
        val b = photographer("w-k$tag")
        val legacyId = "PAY-2001W02-WK${tag.uppercase()}"
        cycles.saveAndFlush(PayoutCycle(id = legacyId, photographerId = a.id, weekOf = monday,
            amountPhp = BigDecimal("111.00"), statusWire = PayoutCycleStatus.PENDING.wire))
        sale(b.id, "900.00", monday.plusDays(1).atStartOfDay(manila).toOffsetDateTime())

        admin.generateForWeek(monday)

        assertEquals(0, BigDecimal("111.00").compareTo(cycles.findById(legacyId).get().amountPhp))
        assertEquals(1, cycles.findAll().count { it.photographerId == b.id })
    }

    @Test
    fun `running the same week twice refreshes the photographer's row instead of adding one`() {
        val monday = LocalDate.of(2001, 1, 15)
        val p = photographer(null)
        sale(p.id, "300.00", monday.plusDays(1).atStartOfDay(manila).toOffsetDateTime())
        admin.generateForWeek(monday)
        sale(p.id, "200.00", monday.plusDays(2).atStartOfDay(manila).toOffsetDateTime())

        admin.generateForWeek(monday)

        val own = cycles.findAll().filter { it.photographerId == p.id }
        assertEquals(1, own.size)
        assertEquals(0, BigDecimal("500.00").compareTo(own.single().amountPhp))
    }

    @Test
    fun `re-running a week made before the new ids refreshes the photographer's own legacy row`() {
        val monday = LocalDate.of(2001, 1, 22)
        val p = photographer("lg$tag")
        val legacyId = "PAY-2001W04-LG${tag.uppercase()}"
        cycles.saveAndFlush(PayoutCycle(id = legacyId, photographerId = p.id, weekOf = monday,
            amountPhp = BigDecimal("111.00"), statusWire = PayoutCycleStatus.PENDING.wire))
        sale(p.id, "400.00", monday.plusDays(1).atStartOfDay(manila).toOffsetDateTime())

        admin.generateForWeek(monday)

        val own = cycles.findAll().filter { it.photographerId == p.id }
        assertEquals(listOf(legacyId), own.map { it.id })
        assertEquals(0, BigDecimal("400.00").compareTo(own.single().amountPhp))
    }

    @Test
    fun `insert refuses an existing id and leaves the row alone`() {
        val owner = photographer("ins$tag")
        val id = "PAY-INS${tag.uppercase()}-001"
        cycles.saveAndFlush(PayoutCycle(id = id, photographerId = owner.id, weekOf = LocalDate.now(),
            amountPhp = BigDecimal("5000.00"), statusWire = PayoutCycleStatus.PAID.wire))
        val other = photographer("other$tag")

        assertFailsWith<org.springframework.dao.DataIntegrityViolationException> {
            cycles.insert(PayoutCycle(id = id, photographerId = other.id, weekOf = LocalDate.now(),
                amountPhp = BigDecimal("1.00"), statusWire = PayoutCycleStatus.PENDING.wire))
        }
        val row = cycles.findById(id).get()
        assertEquals(owner.id, row.photographerId)
        assertEquals(0, BigDecimal("5000.00").compareTo(row.amountPhp))
    }

    @Test
    fun `two photographers without a handle get two cycles`() {
        val p1 = photographer(null)
        val p2 = photographer(null)
        listOf(p1, p2).forEach {
            sale(it.id, "800.00", OffsetDateTime.now().minusDays(10))
            primaryAccount(it.id, "No-handle ${it.id}")
        }

        val first = requests.request(p1.id)
        val second = requests.request(p2.id)

        assertNotEquals(first.id, second.id)
        assertEquals(p1.id, cycles.findById(first.id).get().photographerId)
        assertEquals(p2.id, cycles.findById(second.id).get().photographerId)
    }
}
