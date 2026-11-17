package com.quickpitik.support

import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

// Pins DownloadGrantRepository.recordDownloads (DW-5): a started download adds one to
// download_count and stamps last_downloaded_at, for the named photos of one order only.
class DownloadGrantRecordIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var grants: DownloadGrantRepository
    @Autowired lateinit var tx: TransactionTemplate

    @Test
    fun `recording a download adds one and stamps the time, only for the named photos`() {
        val photographer = users.saveAndFlush(
            User(
                email = "dw5-${UUID.randomUUID()}@test.local",
                passwordHash = "unused",
                name = "Photographer",
                role = Role.PHOTOGRAPHER,
            ),
        )
        val event = events.saveAndFlush(
            Event(
                slug = "dw5-${UUID.randomUUID()}",
                name = "Download count fixture",
                date = LocalDate.of(2026, 10, 5),
                location = "Cebu",
                status = EventStatus.ACTIVE,
            ),
        )
        fun photo() = photos.saveAndFlush(
            Photo(
                eventId = event.id,
                photographerId = photographer.id,
                s3Key = "synthetic/${UUID.randomUUID()}.jpg",
                pricePhp = BigDecimal("125.00"),
            ),
        )
        fun order() = orders.saveAndFlush(
            Order(
                eventId = event.id,
                recipientEmail = "buyer@test.local",
                paymentMethodWire = "qrph",
                status = OrderStatus.PAID,
                totalPhp = BigDecimal("250.00"),
            ),
        )
        val downloaded = photo()
        val notDownloaded = photo()
        val order = order()
        val otherOrder = order()
        val until = OffsetDateTime.now().plusDays(30)
        listOf(downloaded, notDownloaded).forEach { p ->
            grants.saveAndFlush(DownloadGrant(id = DownloadGrantId(order.id, p.id), grantedUntil = until))
        }
        // Same photo, another order: a download on `order` must not count here.
        grants.saveAndFlush(DownloadGrant(id = DownloadGrantId(otherOrder.id, downloaded.id), grantedUntil = until))

        val first = OffsetDateTime.parse("2026-11-09T02:00:00Z")
        val second = OffsetDateTime.parse("2026-11-09T03:30:00Z")
        val changedFirst = tx.execute { grants.recordDownloads(order.id, listOf(downloaded.id), first) }
        val changedSecond = tx.execute { grants.recordDownloads(order.id, listOf(downloaded.id), second) }

        fun grant(o: Order, p: Photo) = grants.findById(DownloadGrantId(o.id, p.id)).orElseThrow()
        assertEquals(1, changedFirst)
        assertEquals(1, changedSecond)
        assertEquals(2, grant(order, downloaded).downloadCount)
        assertEquals(second.toInstant(), grant(order, downloaded).lastDownloadedAt?.toInstant())
        assertEquals(0, grant(order, notDownloaded).downloadCount)
        assertNull(grant(order, notDownloaded).lastDownloadedAt)
        assertEquals(0, grant(otherOrder, downloaded).downloadCount)
        assertNull(grant(otherOrder, downloaded).lastDownloadedAt)
    }
}
