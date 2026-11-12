package com.quickpitik.support

import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import org.springframework.data.domain.Pageable
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Pins DisputeRepository.pageForAdmin: the admin queue's status filter and its `q` search (plan DW-2, 5.2).
// The schema is shared across classes, so every fixture carries its own unique keys.
class DisputeAdminQueryIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var settings: PhotographerSettingsRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var payments: PaymentRepository
    @Autowired lateinit var disputes: DisputeRepository

    private val all = listOf("open", "escalated", "resolved", "denied", "withdrawn")

    private fun search(q: String, statuses: List<String> = all) =
        disputes.pageForAdmin(statuses, q, PageRequest.of(0, 50))

    private fun assertFinds(q: String, vararg expected: Fixture) {
        val page = search(q)
        assertEquals(expected.map { it.dispute.id }, page.content.map { it.id }, "q=$q")
        assertEquals(expected.size.toLong(), page.totalElements, "total for q=$q")
    }

    @Test
    fun `a status list returns only those statuses, newest first`() {
        val tag = token()
        val older = fixture(eventName = "Status $tag A", openedAt = OffsetDateTime.parse("2026-11-01T00:00:00Z"))
        val newer = fixture(eventName = "Status $tag B", status = DisputeStatus.ESCALATED,
            openedAt = OffsetDateTime.parse("2026-11-02T00:00:00Z"))
        fixture(eventName = "Status $tag C", status = DisputeStatus.RESOLVED)

        val page = search(tag, listOf("open", "escalated"))

        assertEquals(listOf(newer.dispute.id, older.dispute.id), page.content.map { it.id })
        assertEquals(2, page.totalElements)
    }

    @Test
    fun `a blank q matches every row`() {
        val a = fixture()
        val b = fixture(status = DisputeStatus.DENIED)

        val page = disputes.pageForAdmin(all, "", Pageable.unpaged())

        assertTrue(page.content.map { it.id }.containsAll(listOf(a.dispute.id, b.dispute.id)))
        assertEquals(disputes.count(), page.totalElements)
    }

    @Test
    fun `q finds a dispute by the runner's name or e-mail, the photographer's handle, the event name, the order id and a PayMongo payment id`() {
        val f = fixture()

        assertFinds(f.runnerName.uppercase().substring(3), f)
        assertFinds(f.runner!!.email.substring(2, 20).uppercase(), f)
        assertFinds(f.handle.substring(1), f)
        assertFinds(f.eventName.lowercase().substring(4), f)
        assertFinds(f.order.id.toString(), f)
        assertFinds(f.payId, f)
        assertFinds(f.checkoutRef, f)
    }

    @Test
    fun `q finds a deleted runner's dispute by the order's recipient e-mail`() {
        val f = fixture(deletedRunner = true)
        val withRunner = fixture()

        assertFinds(f.order.recipientEmail.uppercase(), f)
        // The order's e-mail counts only once the runner is gone; before that the runner's own name and e-mail do.
        assertFinds(withRunner.order.recipientEmail)
    }

    @Test
    fun `q finds a dispute by its id or order id pasted in upper case`() {
        val f = fixture()

        assertFinds(f.dispute.id.toString().uppercase(), f)
        assertFinds(f.order.id.toString().uppercase(), f)
    }

    @Test
    fun `q finds a mixed-case pay_ id exactly as pasted`() {
        val f = fixture()

        assertFinds(f.payId, f)
        assertFinds(f.payId.lowercase())
    }

    @Test
    fun `q does not match another runner's dispute`() {
        val mine = fixture()
        fixture()

        assertFinds(mine.runnerName, mine)
        assertFinds(mine.runner!!.email, mine)
    }

    @Test
    fun `a dispute whose runner was deleted is still listed and searchable by order id`() {
        val f = fixture(deletedRunner = true)

        assertTrue(f.dispute.id in disputes.pageForAdmin(all, "", Pageable.unpaged()).content.map { it.id })
        assertFinds(f.order.id.toString(), f)
    }

    class Fixture(
        val dispute: Dispute,
        val order: Order,
        val runner: User?,
        val runnerName: String,
        val handle: String,
        val eventName: String,
        val payId: String,
        val checkoutRef: String,
    )

    private fun token() = UUID.randomUUID().toString().replace("-", "").take(12)

    private fun fixture(
        status: DisputeStatus = DisputeStatus.OPEN,
        eventName: String = "Dispute search ${token()} Run",
        openedAt: OffsetDateTime = OffsetDateTime.now(),
        deletedRunner: Boolean = false,
    ): Fixture {
        val t = token()
        val runnerName = "Runner Name $t"
        val runner = if (deletedRunner) null else user(Role.RUNNER, runnerName, "dw2-runner-$t@test.local")
        val photographer = user(Role.PHOTOGRAPHER, "Photographer $t", "dw2-photographer-$t@test.local")
        val handle = "dw2-$t"
        settings.saveAndFlush(PhotographerSettings(userId = photographer.id, handle = handle))
        val event = events.saveAndFlush(
            Event(
                slug = "dw2-${UUID.randomUUID()}",
                name = eventName,
                date = LocalDate.of(2026, 10, 15),
                location = "Cebu",
                status = EventStatus.ACTIVE,
            ),
        )
        val photo = photos.saveAndFlush(
            Photo(
                eventId = event.id,
                photographerId = photographer.id,
                s3Key = "dw2/${UUID.randomUUID()}.jpg",
                pricePhp = BigDecimal("100.00"),
            ),
        )
        val order = orders.saveAndFlush(
            Order(
                userId = runner?.id,
                eventId = event.id,
                recipientEmail = "dw2-guest-$t@test.local",
                paymentMethodWire = "qrph",
                status = OrderStatus.FULFILLED,
                totalPhp = BigDecimal("100.00"),
            ),
        )
        val payId = "pay_MiXeD$t"
        val checkoutRef = "cs_CkOuT$t"
        payments.saveAndFlush(
            Payment(
                orderId = order.id,
                provider = "paymongo",
                providerRef = checkoutRef,
                providerPaymentId = payId,
                amountPhp = BigDecimal("100.00"),
                status = PaymentStatus.SUCCEEDED,
            ),
        )
        val dispute = disputes.saveAndFlush(
            Dispute(
                orderId = order.id,
                photoId = photo.id,
                runnerId = runner?.id,
                reasonWire = "other",
                statusWire = status.wire,
                openedAt = openedAt,
            ),
        )
        return Fixture(dispute, order, runner, runnerName, handle, eventName, payId, checkoutRef)
    }

    private fun user(role: Role, name: String, email: String) = users.saveAndFlush(
        User(email = email, passwordHash = "test-only-unused", name = name, role = role),
    )
}
