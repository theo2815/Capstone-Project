package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.entity.Transaction
import com.quickpitik.entity.Dispute
import com.quickpitik.security.JwtTokenProvider
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import java.math.BigDecimal
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PhotographerPurchaseHistoryIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var orders: OrderRepository
    @Autowired private lateinit var items: OrderItemRepository
    @Autowired private lateinit var transactions: TransactionRepository
    @Autowired private lateinit var disputes: DisputeRepository

    @Autowired private lateinit var tokens: JwtTokenProvider
    @LocalServerPort private var port: Int = 0
    private val client = HttpClient.newHttpClient()
    private val json = ObjectMapper()

    @Test
    fun `purchased gallery and history use settled item identities and private money`() {
        val owner = seller()
        val other = seller()
        val buyer = users.save(User(email = "aira.${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Aira Santos", role = Role.RUNNER))
        val event = event(EventStatus.ACTIVE, owner.id)
        coverage.save(EventPhotographer(EventPhotographerId(event.id, owner.id)))
        val photo = photos.save(Photo(eventId = event.id, photographerId = owner.id, s3Key = "history/sold.jpg", pricePhp = BigDecimal("999.00")))
        val otherPhoto = photos.save(Photo(eventId = event.id, photographerId = other.id, s3Key = "history/other.jpg", pricePhp = BigDecimal("999.00")))
        val paidAt = OffsetDateTime.parse("2026-09-01T10:00:00Z")
        fun purchase(status: OrderStatus, price: String, discount: String, paid: OffsetDateTime? = paidAt): Order {
            val order = orders.save(Order(userId = buyer.id, eventId = event.id, recipientEmail = buyer.email, paymentMethodWire = "gcash", status = status, totalPhp = BigDecimal("900.00"), paidAt = paid))
            items.save(OrderItem(OrderItemId(order.id, photo.id), BigDecimal(price), BigDecimal(discount)))
            return order
        }
        val paid = purchase(OrderStatus.FULFILLED, "100.00", "20.00")
        items.save(OrderItem(OrderItemId(paid.id, otherPhoto.id), BigDecimal("800.00")))
        val free = purchase(OrderStatus.FULFILLED, "100.00", "100.00")
        val refunded = purchase(OrderStatus.REFUNDED, "100.00", "0.00")
        purchase(OrderStatus.PENDING, "100.00", "0.00", null)
        transactions.save(Transaction(paidAt = paidAt, photographerId = owner.id, eventId = event.id, photoId = photo.id, orderId = paid.id, buyerId = buyer.id, buyerDisplayName = "Aira S.", amountKeptPhp = BigDecimal("55.00")))
        val sale = transactions.save(Transaction(paidAt = paidAt, photographerId = owner.id, eventId = event.id, photoId = photo.id, orderId = refunded.id, buyerDisplayName = "Aira S.", amountKeptPhp = BigDecimal("75.00")))
        val refundCreatedAt = paidAt.plusDays(2)
        val refund = transactions.save(Transaction(paidAt = paidAt, photographerId = owner.id, eventId = event.id, photoId = photo.id, orderId = refunded.id, buyerDisplayName = "Aira S.", amountKeptPhp = BigDecimal("-75.00"), isRefund = true, refundOf = sale.id, createdAt = refundCreatedAt))
        disputes.save(Dispute(orderId = refunded.id, photoId = photo.id, reasonWire = "other", statusWire = "resolved", resolutionWire = "refund_full", refundStatus = "succeeded", refundAmountPhp = BigDecimal("100.00"), refundedAt = paidAt.plusDays(1)))

        val token = tokens.createAccessToken(owner)
        val galleryResponse = get("/api/v1/me/photographer/events/${event.id}/purchased-photos", token)
        assertEquals(200, galleryResponse.statusCode())
        val gallery = json.readTree(galleryResponse.body()).path("data")
        assertEquals(1, gallery.path("total").asInt())
        assertEquals(3, gallery.path("items")[0].path("purchaseCount").asInt())
        val historyResponse = get("/api/v1/me/photographer/photos/${photo.id}/purchases?orderId=${paid.id}", token)
        assertEquals(200, historyResponse.statusCode())
        val history = json.readTree(historyResponse.body()).path("data")
        assertEquals(3, history.path("total").asInt())
        val selected = history.path("selectedPurchase")
        assertEquals(0, BigDecimal("80.00").compareTo(selected.path("paidPhp").decimalValue()))
        assertEquals(0, BigDecimal("55.00").compareTo(selected.path("netKeptPhp").decimalValue()))
        assertEquals("Aira S.", selected.path("buyer").asText())
        val rows = history.path("items").associateBy { it.path("orderId").asText() }
        assertEquals("free", rows.getValue(free.id.toString()).path("kind").asText())
        assertEquals(0, BigDecimal.ZERO.compareTo(rows.getValue(free.id.toString()).path("netKeptPhp").decimalValue()))
        assertEquals("full", rows.getValue(refunded.id.toString()).path("refundStatus").asText())
        assertEquals(2, rows.getValue(refunded.id.toString()).path("transactions").size())
        for (secret in listOf("recipientEmail", "buyerId", "userId", "totalPhp", "shareToken", otherPhoto.id.toString())) {
            assertFalse(historyResponse.body().contains(secret), secret)
        }
        assertEquals(404, get("/api/v1/me/photographer/photos/${otherPhoto.id}/purchases", token).statusCode())
        assertEquals(404, get("/api/v1/me/photographer/photos/${photo.id}/purchases?orderId=${UUID.randomUUID()}", token).statusCode())
        val runnerToken = tokens.createAccessToken(buyer)
        assertEquals(403, get("/api/v1/me/photographer/events/${event.id}/purchased-photos", runnerToken).statusCode())
        assertEquals(403, get("/api/v1/me/photographer/photos/${photo.id}/purchases", runnerToken).statusCode())
        assertEquals(401, get("/api/v1/me/photographer/photos/${photo.id}/purchases").statusCode())

        val earnings = json.readTree(get("/api/v1/me/photographer/earnings/per-event?limit=200", token).body()).path("data")
        val earning = earnings.path("items").single { it.path("eventId").asText() == event.id.toString() }
        assertEquals(3, earning.path("purchaseCount").asInt())
        assertTrue(earning.path("deletedAt").isNull)

        val ledger = json.readTree(get("/api/v1/me/photographer/billing/transactions?limit=50", token).body()).path("data")
        val refundRow = ledger.path("items").single { it.path("id").asText() == refund.id.toString() }
        assertEquals(refunded.id.toString(), refundRow.path("orderId").asText())
        assertTrue(refundRow.path("isRefund").asBoolean())
        assertEquals(sale.id.toString(), refundRow.path("refundOf").asText())
        assertEquals(paidAt, OffsetDateTime.parse(refundRow.path("paidAt").asText()))
        assertEquals(refundCreatedAt, OffsetDateTime.parse(refundRow.path("recordedAt").asText()))
        assertFalse(refundRow.path("thumbnailUrl").isNull)
    }

    @Test
    fun `history pages tied purchases and pins an order outside first page`() {
        val owner = seller()
        val event = event(EventStatus.ACTIVE, owner.id)
        val photo = photos.save(Photo(eventId = event.id, photographerId = owner.id, s3Key = "history/paged.jpg", pricePhp = BigDecimal("10.00")))
        val at = OffsetDateTime.parse("2026-09-01T10:00:00Z")
        val orderIds = (0 until 51).map {
            val order = orders.save(Order(eventId = event.id, recipientEmail = "guest@test.local", paymentMethodWire = "gcash", status = OrderStatus.FULFILLED, totalPhp = BigDecimal("10.00"), paidAt = at))
            items.save(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("10.00")))
            order.id
        }.sortedBy { it.toString() }
        val token = tokens.createAccessToken(owner)
        val path = "/api/v1/me/photographer/photos/${photo.id}/purchases"
        val first = get("$path?limit=50&orderId=${orderIds.last()}", token)
        assertEquals(200, first.statusCode())
        val firstData = json.readTree(first.body()).path("data")
        assertEquals(51, firstData.path("total").asInt())
        assertEquals(orderIds.last().toString(), firstData.path("selectedPurchase").path("orderId").asText())
        assertEquals(orderIds.first().toString(), firstData.path("items")[0].path("orderId").asText())
        val second = json.readTree(get("$path?offset=50&limit=1", token).body()).path("data")
        assertEquals(orderIds.last().toString(), second.path("items")[0].path("orderId").asText())
    }

    @Test
    fun `removed and purging photos retain private receipts without an image`() {
        val owner = seller()
        val foreign = seller()
        val removedAt = OffsetDateTime.parse("2026-09-04T21:36:27Z")
        val removedEvent = event(EventStatus.ARCHIVED, owner.id, removedAt)
        val photo = photos.save(Photo(
            eventId = removedEvent.id, photographerId = owner.id, s3Key = "history/purging.jpg",
            pricePhp = BigDecimal("100.00"), deletedAt = removedAt, deletionStartedAt = removedAt,
        ))
        val hidden = photos.save(Photo(
            eventId = removedEvent.id, photographerId = owner.id, s3Key = "history/hidden.jpg",
            pricePhp = BigDecimal("100.00"), status = PhotoStatus.HIDDEN,
        ))
        val trashed = photos.save(Photo(
            eventId = removedEvent.id, photographerId = owner.id, s3Key = "history/trashed.jpg",
            pricePhp = BigDecimal("100.00"), deletedAt = removedAt,
        ))
        val order = orders.save(Order(
            eventId = removedEvent.id, recipientEmail = "guest.runner@test.local", paymentMethodWire = "gcash",
            status = OrderStatus.FULFILLED, totalPhp = BigDecimal("300.00"), paidAt = removedAt,
        ))
        items.save(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("100.00")))
        items.save(OrderItem(OrderItemId(order.id, hidden.id), BigDecimal("100.00")))
        items.save(OrderItem(OrderItemId(order.id, trashed.id), BigDecimal("100.00")))
        val token = tokens.createAccessToken(owner)
        val gallery = json.readTree(get("/api/v1/me/photographer/events/${removedEvent.id}/purchased-photos", token).body()).path("data")
        assertEquals(3, gallery.path("total").asInt())
        val galleryRows = gallery.path("items").associateBy { it.path("photo").path("id").asText() }
        assertTrue(galleryRows.getValue(photo.id.toString()).path("photo").path("thumbnailUrl").isNull)
        assertEquals("hidden", galleryRows.getValue(hidden.id.toString()).path("photo").path("status").asText())
        assertEquals(200, get("/api/v1/me/photographer/photos/${hidden.id}/purchases", token).statusCode())
        assertEquals(200, get("/api/v1/me/photographer/photos/${trashed.id}/purchases", token).statusCode())
        val history = json.readTree(get("/api/v1/me/photographer/photos/${photo.id}/purchases", token).body()).path("data")
        assertEquals(1, history.path("total").asInt())
        assertEquals(removedAt.toString(), history.path("event").path("deletedAt").asText())
        assertEquals("Guest R.", history.path("items")[0].path("buyer").asText())
        assertEquals(404, get("/api/v1/me/photographer/photos/${photo.id}/purchases", tokens.createAccessToken(foreign)).statusCode())

        val emptyEvent = event(EventStatus.ACTIVE, owner.id)
        val emptyPhoto = photos.save(Photo(eventId = emptyEvent.id, photographerId = owner.id, s3Key = "history/empty.jpg", pricePhp = BigDecimal("0.00")))
        val emptyGallery = json.readTree(get("/api/v1/me/photographer/events/${emptyEvent.id}/purchased-photos", token).body()).path("data")
        assertEquals(0, emptyGallery.path("total").asInt())
        val emptyHistory = json.readTree(get("/api/v1/me/photographer/photos/${emptyPhoto.id}/purchases", token).body()).path("data")
        assertEquals(0, emptyHistory.path("total").asInt())
    }

    @Test
    fun `pending refund and missing ledger never invent refunded or kept money`() {
        val owner = seller()
        val buyer = users.save(User(email = "buyer.${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Aira Santos", role = Role.RUNNER))
        val event = event(EventStatus.ACTIVE, owner.id)
        val photo = photos.save(Photo(eventId = event.id, photographerId = owner.id, s3Key = "history/refund.jpg", pricePhp = BigDecimal("100.00")))
        val at = OffsetDateTime.parse("2026-09-01T10:00:00Z")
        fun order(): Order {
            val saved = orders.save(Order(userId = buyer.id, eventId = event.id, recipientEmail = buyer.email, paymentMethodWire = "card", status = OrderStatus.FULFILLED, totalPhp = BigDecimal("100.00"), paidAt = at))
            items.save(OrderItem(OrderItemId(saved.id, photo.id), BigDecimal("100.00")))
            return saved
        }
        val pending = order()
        val missingLedger = order()
        transactions.save(Transaction(paidAt = at, photographerId = owner.id, eventId = event.id, photoId = photo.id, orderId = pending.id, buyerId = buyer.id, amountKeptPhp = BigDecimal("75.00")))
        disputes.save(Dispute(orderId = pending.id, photoId = photo.id, reasonWire = "other", refundStatus = "pending", refundAmountPhp = BigDecimal("100.00")))
        val token = tokens.createAccessToken(owner)
        val path = "/api/v1/me/photographer/photos/${photo.id}/purchases"
        val pendingPurchase = json.readTree(get("$path?orderId=${pending.id}", token).body()).path("data").path("selectedPurchase")
        assertEquals("Aira S.", pendingPurchase.path("buyer").asText())
        assertEquals("pending", pendingPurchase.path("refundStatus").asText())
        assertTrue(pendingPurchase.path("refundPending").asBoolean())
        assertEquals(0, BigDecimal.ZERO.compareTo(pendingPurchase.path("refundedPhp").decimalValue()))
        val noLedger = json.readTree(get("$path?orderId=${missingLedger.id}", token).body()).path("data").path("selectedPurchase")
        assertTrue(noLedger.path("netKeptPhp").isNull)
    }

    @Test
    fun `partial legacy recorded and denied refunds use only completed item facts`() {
        val owner = seller()
        val event = event(EventStatus.ACTIVE, owner.id)
        val photo = photos.save(Photo(eventId = event.id, photographerId = owner.id, s3Key = "history/refund-cases.jpg", pricePhp = BigDecimal("100.00")))
        val at = OffsetDateTime.parse("2026-09-01T10:00:00Z")
        fun purchase(): Order {
            val order = orders.save(Order(eventId = event.id, recipientEmail = "guest@test.local", paymentMethodWire = "gcash", status = OrderStatus.FULFILLED, totalPhp = BigDecimal("100.00"), paidAt = at))
            items.save(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("100.00")))
            return order
        }
        val partial = purchase()
        val legacy = purchase()
        val recorded = purchase()
        val denied = purchase()
        disputes.save(Dispute(orderId = partial.id, photoId = photo.id, reasonWire = "other", statusWire = "resolved", resolutionWire = "refund_partial", refundStatus = "succeeded", refundAmountPhp = BigDecimal("25.00"), refundedAt = at.plusDays(1)))
        disputes.save(Dispute(orderId = legacy.id, photoId = photo.id, reasonWire = "other", statusWire = "resolved", resolutionWire = "refund_partial", refundAmountPhp = BigDecimal("10.00"), refundedAt = at.plusDays(2)))
        disputes.save(Dispute(orderId = denied.id, photoId = photo.id, reasonWire = "other", statusWire = "denied", refundStatus = "failed", refundAmountPhp = BigDecimal("100.00")))
        transactions.save(Transaction(paidAt = at, photographerId = owner.id, eventId = event.id, photoId = photo.id, orderId = recorded.id, amountKeptPhp = BigDecimal("-25.00"), isRefund = true))

        val response = get("/api/v1/me/photographer/photos/${photo.id}/purchases", tokens.createAccessToken(owner))
        assertEquals(200, response.statusCode())
        val rows = json.readTree(response.body()).path("data").path("items").associateBy { it.path("orderId").asText() }
        assertEquals(4, rows.size)
        for ((order, amount) in listOf(partial to "25.00", legacy to "10.00")) {
            val row = rows.getValue(order.id.toString())
            assertEquals("partial", row.path("refundStatus").asText())
            assertEquals(0, BigDecimal(amount).compareTo(row.path("refundedPhp").decimalValue()))
        }
        assertEquals("recorded", rows.getValue(recorded.id.toString()).path("refundStatus").asText())
        assertTrue(rows.getValue(recorded.id.toString()).path("refundedPhp").isNull)
        val deniedRow = rows.getValue(denied.id.toString())
        assertEquals("none", deniedRow.path("refundStatus").asText())
        assertFalse(deniedRow.path("refundPending").asBoolean())
        assertEquals(0, BigDecimal.ZERO.compareTo(deniedRow.path("refundedPhp").decimalValue()))
    }

    @Test
    fun `removed event is visible only through an authorized history read`() {
        val owner = seller()
        val foreign = seller()
        val legacySeller = seller()
        val removedAt = OffsetDateTime.parse("2026-09-04T21:36:27Z")
        val event = event(EventStatus.ARCHIVED, owner.id, removedAt)
        coverage.save(EventPhotographer(EventPhotographerId(event.id, owner.id), photoCount = 1))
        photos.save(Photo(eventId = event.id, photographerId = owner.id, s3Key = "history/owner.jpg", pricePhp = BigDecimal("100.00")))
        val legacyPhoto = photos.save(Photo(eventId = event.id, photographerId = legacySeller.id, s3Key = "history/legacy.jpg", pricePhp = BigDecimal("100.00")))
        val order = orders.save(Order(eventId = event.id, recipientEmail = "buyer@test.local", paymentMethodWire = "card", status = OrderStatus.FULFILLED, totalPhp = BigDecimal("100.00")))
        items.save(OrderItem(OrderItemId(order.id, legacyPhoto.id), BigDecimal("100.00")))

        val path = "/api/v1/me/photographer/events/${event.id}"
        val ownerToken = tokens.createAccessToken(owner)
        val foreignToken = tokens.createAccessToken(foreign)
        val legacyToken = tokens.createAccessToken(legacySeller)
        val history = get("$path?includeRemoved=true", ownerToken)
        assertEquals(200, history.statusCode())
        assertEquals(removedAt.toString(), json.readTree(history.body()).path("data").path("deletedAt").asText())
        assertEquals(404, get(path, ownerToken).statusCode())
        assertEquals(404, get("$path?includeRemoved=true", foreignToken).statusCode())
        assertEquals(200, get("$path?includeRemoved=true", legacyToken).statusCode())
        assertEquals(404, get(path, legacyToken).statusCode())
        assertEquals(404, get("/api/v1/me/photographer/events/${UUID.randomUUID()}?includeRemoved=true", ownerToken).statusCode())
        assertEquals(404, get("/api/v1/events/${event.slug}").statusCode())
        assertEquals(404, get("$path/photos", ownerToken).statusCode())
        assertEquals(404, get("$path/photos?deleted=true", ownerToken).statusCode())
    }

    @Test
    fun `removed event rejects photo mutations while active and archived details remain readable`() {
        val owner = seller()
        val token = tokens.createAccessToken(owner)
        for (status in listOf(EventStatus.ACTIVE, EventStatus.ARCHIVED)) {
            val available = event(status, owner.id)
            assertEquals(200, get("/api/v1/me/photographer/events/${available.id}", token).statusCode())
        }
        val removed = event(EventStatus.ACTIVE, owner.id, OffsetDateTime.parse("2026-09-04T21:36:27Z"))
        val photo = photos.save(Photo(eventId = removed.id, photographerId = owner.id, s3Key = "history/mutation.jpg", pricePhp = BigDecimal("100.00")))
        val body = """{"photoIds":["${photo.id}"]}"""
        val base = "/api/v1/me/photographer/events/${removed.id}/photos"
        for (action in listOf("trash", "restore", "permanent-delete")) {
            assertEquals(404, post("$base/$action", body, token).statusCode(), action)
        }
    }

    private fun seller() = users.save(User(email = "history-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Seller", role = Role.PHOTOGRAPHER))

    private fun event(status: EventStatus, owner: UUID, deletedAt: OffsetDateTime? = null) = events.save(
        Event(slug = "history-${UUID.randomUUID()}", name = "History", date = LocalDate.of(2026, 9, 1), location = "Cebu", status = status, createdBy = owner, deletedAt = deletedAt),
    )

    private fun get(path: String, token: String? = null): HttpResponse<String> = send(HttpRequest.newBuilder(uri(path)).GET(), token)

    private fun post(path: String, body: String, token: String): HttpResponse<String> = send(
        HttpRequest.newBuilder(uri(path)).header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)), token,
    )

    private fun send(builder: HttpRequest.Builder, token: String?): HttpResponse<String> {
        token?.let { builder.header("Authorization", "Bearer $it") }
        return client.send(builder.build(), HttpResponse.BodyHandlers.ofString())
    }

    private fun uri(path: String) = URI("http://localhost:$port$path")
}
