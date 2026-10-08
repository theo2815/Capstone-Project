package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.orders.*
import com.quickpitik.service.photographer.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import com.quickpitik.service.removal.PhotoRemovalQueryService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertAll
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.http.HttpStatus
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionTemplate
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.math.BigDecimal
import java.net.URI
import java.net.http.*
import java.time.*
import java.util.UUID
import kotlin.test.*

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PhotoRemovalDeliveryIntegrationTest : PostgresIntegrationTest() {
    @LocalServerPort var port = 0
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var reviews: PhotoRemovalReviewRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var orderItems: OrderItemRepository
    @Autowired lateinit var payments: PaymentRepository
    @Autowired lateinit var transactions: TransactionRepository
    @Autowired lateinit var grants: DownloadGrantRepository
    @Autowired lateinit var availability: PhotoAvailabilityService
    @Autowired lateinit var orderService: OrderService
    @Autowired lateinit var bundles: OrderBundleService
    @Autowired lateinit var ownerBundles: PhotographerBundleService
    @Autowired lateinit var ownerEvents: PhotographerEventService
    @Autowired lateinit var photoMapper: PhotographerPhotoDtoMapper
    @Autowired lateinit var metadata: PhotoMetadataService
    @Autowired lateinit var webhook: PaymongoWebhookService
    @Autowired lateinit var receipt: OrderReceiptEmailService
    @Autowired lateinit var tokens: OrderAccessTokenService
    @Autowired lateinit var removalNotices: OrderPhotoRemovalService
    @Autowired lateinit var inbox: RunnerMessageRepository
    @Autowired lateinit var historyRecorder: com.quickpitik.service.removal.PhotoRemovalHistoryRecorder
    @Autowired lateinit var mapper: com.fasterxml.jackson.databind.ObjectMapper
    @Autowired lateinit var queries: PhotoRemovalQueryService
    @Autowired lateinit var purchases: com.quickpitik.service.earnings.PhotographerPurchaseService
    @Autowired lateinit var ledgerReader: com.quickpitik.service.earnings.TransactionsLedgerService
    @Autowired lateinit var messageReader: com.quickpitik.service.runner.RunnerMessageReaderService
    @Autowired lateinit var publicPhotoController: com.quickpitik.controller.EventPhotoController
    @Autowired lateinit var publicPhotos: com.quickpitik.service.photos.PhotoService
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var jdbc: JdbcTemplate
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var resend: com.quickpitik.service.email.ResendClient
    @MockitoBean lateinit var paymongo: PaymongoClient
    private lateinit var owner: User
    private lateinit var buyer: User
    private lateinit var event: Event
    private lateinit var photo: Photo
    private lateinit var item: PhotoRemovalItem
    private val bytes = byteArrayOf(1, 2, 3)

    @BeforeEach fun fixture() {
        owner = users.saveAndFlush(User(email = "owner-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Owner", role = Role.PHOTOGRAPHER))
        buyer = users.saveAndFlush(User(email = "buyer-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Buyer", role = Role.RUNNER))
        event = events.saveAndFlush(Event(slug = "delivery-${UUID.randomUUID()}", name = "Cebu run", date = LocalDate.of(2026, 10, 7), location = "Cebu", status = EventStatus.ACTIVE))
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "synthetic/${UUID.randomUUID()}.jpg",
            thumbnailS3Key = "synthetic/thumb-${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("125.00")))
        val request = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id, eventIdSnapshot = event.id, eventNameSnapshot = event.name,
            requesterContact = "Private contact", supportReference = "Case-${UUID.randomUUID()}", requesterReason = "Privacy",
            publicToOwnerReason = "Privacy request", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(), status = RemovalRequestStatus.OPEN))
        val review = reviews.saveAndFlush(PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = owner.id, ownerId = owner.id))
        item = items.saveAndFlush(PhotoRemovalItem(requestId = request.id, photoIdSnapshot = photo.id, photoId = photo.id,
            ownerIdSnapshot = owner.id, reviewId = review.id, state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL))
        Mockito.`when`(storage.open(anyArg())).thenAnswer { ByteArrayInputStream(bytes) }
        Mockito.`when`(storage.getBytes(anyArg())).thenReturn(bytes)
        Mockito.`when`(storage.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://synthetic.test/image")
        Mockito.`when`(storage.presignedDownloadUrl(anyArg(), anyArg(), anyArg())).thenReturn("https://synthetic.test/original")
        Mockito.`when`(storage.exists(anyArg())).thenReturn(true)
        Mockito.`when`(resend.send(anyArg())).thenReturn(com.quickpitik.dto.email.ResendSendEmailResponse("re_synthetic"))
    }
    private fun <T> anyArg(): T = Mockito.any()
    private fun order(pending: Boolean = false, guest: Boolean = false): Order {
        val order = orders.saveAndFlush(Order(userId = if (guest) null else buyer.id, eventId = event.id,
            recipientEmail = buyer.email, paymentMethodWire = "gcash", totalPhp = BigDecimal("125.00"),
            status = if (pending) OrderStatus.PENDING else OrderStatus.PAID, paidAt = if (pending) null else OffsetDateTime.now()))
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("125.00")))
        if (!pending) grants.saveAndFlush(DownloadGrant(DownloadGrantId(order.id, photo.id), OffsetDateTime.now().plusYears(1)))
        return order
    }
    private fun authorize() {
        val actor = UUID.randomUUID()
        item.state = RemovalItemState.REMOVAL_AUTHORIZED
        item.adminDecision = RemovalDecision.REMOVE
        item.adminDecisionReason = "Verified privacy request"
        item.adminDecisionActorId = actor
        item.adminDecidedAt = Instant.now()
        items.saveAndFlush(item)
        tx.executeWithoutResult { availability.authorizeRemoval(photo.id, item.id, actor) }
    }
    private fun awaitReceiptDelivery(orderId: UUID) {
        val deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(10)
        while (System.nanoTime() < deadline) {
            if (orders.findById(orderId).orElseThrow().receiptDeliveredAt != null) return
            Thread.sleep(20)
        }
        error("Receipt listener did not complete before financial baseline")
    }
    private fun download(order: Order): HttpResponse<ByteArray> {
        val token = tokens.issue(order, OrderCapability.BUNDLE)
        val request = HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/orders/${order.id}/download-bundle?token=$token&photo=${photo.id}"))
            .timeout(Duration.ofSeconds(10)).GET().build()
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofByteArray())
    }
    @Test fun `prepared free response cannot open an original after authorized removal`() {
        event.pricingMode = EventPricingMode.FREE
        events.saveAndFlush(event)
        val response = publicPhotoController.downloadFree(event.id, photo.id, org.springframework.mock.web.MockHttpServletRequest())
        assertEquals(HttpStatus.OK, response.statusCode)
        authorize()
        val out = ByteArrayOutputStream()
        val failure = assertFailsWith<com.quickpitik.exception.NotFoundException> { response.body!!.writeTo(out) }
        assertEquals(com.quickpitik.common.ErrorCodes.PHOTO_NOT_FOUND, failure.code)
        assertEquals(0, out.size())
        Mockito.verify(storage, Mockito.never()).open(anyArg())
    }
    @Test fun `prepared free response cannot open an original after review restricts public access`() {
        event.pricingMode = EventPricingMode.FREE
        events.saveAndFlush(event)
        val response = publicPhotoController.downloadFree(event.id, photo.id, org.springframework.mock.web.MockHttpServletRequest())
        assertEquals(HttpStatus.OK, response.statusCode)
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        assertEquals(RemovalState.REVIEW, photos.findCurrentRemovalState(photo.id))
        val out = ByteArrayOutputStream()
        val failure = assertFailsWith<com.quickpitik.exception.NotFoundException> { response.body!!.writeTo(out) }
        assertEquals(com.quickpitik.common.ErrorCodes.PHOTO_NOT_FOUND, failure.code)
        assertEquals(0, out.size())
        Mockito.verify(storage, Mockito.never()).open(anyArg())
    }
    @Test fun `free streaming uses current event and catalog eligibility outside the database transaction`() {
        event.pricingMode = EventPricingMode.FREE
        event.status = EventStatus.COMPLETED
        events.saveAndFlush(event)
        val download = publicPhotos.freeDownload(event.id, photo.id)
        Mockito.`when`(storage.open(anyArg())).thenAnswer {
            assertFalse(org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive())
            ByteArrayInputStream(bytes)
        }
        val out = ByteArrayOutputStream()
        tx.executeWithoutResult { publicPhotos.writeFreeDownload(event.id, photo.id, download, out) }
        assertTrue(bytes.contentEquals(out.toByteArray()))
        fun denied(eventId: UUID = event.id) {
            Mockito.clearInvocations(storage)
            val blocked = ByteArrayOutputStream()
            val failure = assertFailsWith<com.quickpitik.exception.NotFoundException> {
                publicPhotos.writeFreeDownload(eventId, photo.id, download, blocked)
            }
            assertEquals(com.quickpitik.common.ErrorCodes.PHOTO_NOT_FOUND, failure.code)
            assertEquals(0, blocked.size())
            Mockito.verify(storage, Mockito.never()).open(anyArg())
        }
        denied(UUID.randomUUID())
        event.pricingMode = EventPricingMode.PAID
        events.saveAndFlush(event)
        denied()
        event.pricingMode = EventPricingMode.FREE
        for (status in listOf(EventStatus.DRAFT, EventStatus.ARCHIVED)) {
            event.status = status
            events.saveAndFlush(event)
            denied()
        }
        event.status = EventStatus.ACTIVE
        events.saveAndFlush(event)
        for ((restrict, restore) in listOf(
            "status = 'HIDDEN'" to "status = 'LIVE'",
            "deleted_at = CURRENT_TIMESTAMP" to "deleted_at = NULL",
            "deleted_at = CURRENT_TIMESTAMP, deletion_started_at = CURRENT_TIMESTAMP" to "deletion_started_at = NULL, deleted_at = NULL",
            "purged_at = CURRENT_TIMESTAMP" to "purged_at = NULL",
        )) {
            jdbc.update("UPDATE photos SET $restrict WHERE id = ?", photo.id)
            denied()
            jdbc.update("UPDATE photos SET $restore WHERE id = ?", photo.id)
        }
    }
    @Test fun `buyer notice is durable once per order photo despite removal replay and soft deletion`() {
        val order = order()
        authorize()
        tx.executeWithoutResult { removalNotices.onRemovalAuthorized(photo.id, item.requestId, Instant.now()) }
        val first = inbox.findAll().filter { it.orderId == order.id }
        assertEquals(1, first.size)
        assertEquals("photo_removed", first.single().kindWire)
        first.single().removedAt = OffsetDateTime.now()
        first.single().readAt = OffsetDateTime.now()
        inbox.saveAndFlush(first.single())
        val another = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id, eventIdSnapshot = event.id,
            eventNameSnapshot = event.name, requesterContact = "Private contact", supportReference = "Other-${UUID.randomUUID()}",
            requesterReason = "Privacy", publicToOwnerReason = "Privacy", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID()))
        items.saveAndFlush(PhotoRemovalItem(requestId = another.id, photoIdSnapshot = photo.id, photoId = photo.id,
            ownerIdSnapshot = owner.id, state = RemovalItemState.REMOVAL_AUTHORIZED, matchSource = RemovalMatchSource.MANUAL))
        tx.executeWithoutResult {
            removalNotices.onRemovalAuthorized(photo.id, item.requestId, Instant.now())
            removalNotices.onRemovalAuthorized(photo.id, another.id, Instant.now())
        }
        assertEquals(first.single().id, inbox.findAll().single { it.orderId == order.id }.id)
    }
    @Test fun `guest task is durable idempotent admin only and completed by exact support update`() {
        val order = order(guest = true)
        authorize()
        repeat(2) { tx.executeWithoutResult { removalNotices.onRemovalAuthorized(photo.id, item.requestId, Instant.now()) } }
        val overlap = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id, eventIdSnapshot = event.id,
            eventNameSnapshot = event.name, requesterContact = "Private contact", supportReference = "Overlap-${UUID.randomUUID()}",
            requesterReason = "Privacy", publicToOwnerReason = "Privacy", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID()))
        items.saveAndFlush(PhotoRemovalItem(requestId = overlap.id, photoIdSnapshot = photo.id, photoId = photo.id,
            ownerIdSnapshot = owner.id, state = RemovalItemState.REMOVAL_AUTHORIZED, matchSource = RemovalMatchSource.MANUAL))
        tx.executeWithoutResult { removalNotices.onRemovalAuthorized(photo.id, overlap.id, Instant.now()) }
        assertEquals(1, jdbc.queryForObject("SELECT COUNT(*) FROM photo_removal_history WHERE metadata->>'orderId' = ?", Int::class.java, order.id.toString()))
        val ownerView = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(queries.photographerDetail(owner.id, item.requestId))
        assertFalse(ownerView.has("guestBuyerRemovalNotices"))
        assertFalse(ownerView.toString().contains(order.recipientEmail))
        val notices = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(queries.adminDetail(UUID.randomUUID(), item.requestId))["guestBuyerRemovalNotices"]
        assertNotNull(notices)
        assertEquals(1, notices.size())
        val notice = notices.single()
        assertEquals(order.id.toString(), notice["orderId"].asText())
        assertEquals(photo.id.toString(), notice["photoId"].asText())
        assertEquals(order.recipientEmail, notice["recipientEmail"].asText())
        assertEquals("PENDING", notice["status"].asText())
        tx.executeWithoutResult { historyRecorder.record(com.quickpitik.service.removal.RemovalHistoryEvent(
            requestId = item.requestId, actorType = RemovalActorType.SYSTEM, action = "SUPPORT_UPDATED", occurredAt = Instant.now(),
            commandId = UUID.randomUUID(), metadata = mapOf("noticeId" to UUID.randomUUID().toString(),
                "orderId" to order.id.toString(), "photoId" to photo.id.toString()))) }
        assertEquals("PENDING", mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(queries.adminDetail(UUID.randomUUID(), item.requestId))["guestBuyerRemovalNotices"].single()["status"].asText())
        tx.executeWithoutResult { historyRecorder.record(com.quickpitik.service.removal.RemovalHistoryEvent(
            requestId = item.requestId, actorType = RemovalActorType.SYSTEM, action = "SUPPORT_UPDATED", occurredAt = Instant.now(),
            commandId = UUID.randomUUID(), metadata = mapOf("noticeId" to notice["noticeId"].asText(),
                "orderId" to order.id.toString(), "photoId" to photo.id.toString()))) }
        val completed = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(queries.adminDetail(UUID.randomUUID(), item.requestId))["guestBuyerRemovalNotices"].single()
        assertEquals("COMPLETED", completed["status"].asText())
        assertEquals(0, jdbc.queryForObject("SELECT COUNT(*) FROM runner_messages WHERE order_id = ?", Int::class.java, order.id))
    }
    @Test fun `late settled guest gets followup while failed transaction rolls notices back`() {
        val order = order(pending = true, guest = true)
        val ref = "pi_${UUID.randomUUID()}"
        val payment = payments.saveAndFlush(Payment(orderId = order.id, provider = "paymongo", providerRef = ref, amountPhp = BigDecimal("125.00")))
        authorize()
        assertFailsWith<IllegalStateException> { tx.executeWithoutResult {
            webhook.settlePaymentIntent(ref, "pay_synthetic")
            throw IllegalStateException("Force settlement rollback")
        } }
        assertEquals(PaymentStatus.PENDING, payments.findById(payment.id).orElseThrow().status)
        assertEquals(0, jdbc.queryForObject("SELECT COUNT(*) FROM photo_removal_history WHERE metadata->>'orderId' = ?", Int::class.java, order.id.toString()))
        webhook.settlePaymentIntent(ref, "pay_synthetic")
        val notices = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(queries.adminDetail(UUID.randomUUID(), item.requestId))["guestBuyerRemovalNotices"]
        assertNotNull(notices)
        assertEquals(1, notices.size())
        assertEquals(OrderStatus.FULFILLED, orders.findById(order.id).orElseThrow().status)
        assertEquals(1, grants.findByIdOrderId(order.id).size)
        assertEquals(410, download(order).statusCode())
    }
    @Test fun `owner snapshot cannot sign preview after concurrent authorization`() {
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        var removed = false
        Mockito.`when`(storage.exists(anyArg())).thenAnswer {
            if (!removed) { removed = true; authorize() }
            true
        }
        val detail = queries.photographerDetail(owner.id, item.requestId)
        assertNull(detail.items.single().photo?.imageUrl)
        Mockito.verify(storage, Mockito.never()).presignedGetUrl(anyArg(), anyArg())
    }
    @Test fun `all removed owner photos cannot issue a fresh bundle capability`() {
        authorize()
        assertEquals(RemovalState.REMOVED, photos.findCurrentRemovalState(photo.id))
        assertFalse(photos.hasDeliverableOwnerPhoto(event.id, owner.id))
        val error = assertFailsWith<ApiException> { ownerBundles.prepareDownloadUrl(owner.id, event.id) }
        assertEquals(HttpStatus.GONE, error.status)
    }
    @Test fun `removed purchases still receive financial receipt without promising unavailable originals`() {
        val order = order()
        authorize()
        var sent: com.quickpitik.dto.email.ResendSendEmailRequest? = null
        Mockito.`when`(resend.send(anyArg())).thenAnswer {
            sent = it.getArgument(0)
            com.quickpitik.dto.email.ResendSendEmailResponse("re_synthetic")
        }
        receipt.sendReceiptIfPending(order.id)
        assertNotNull(sent)
        assertTrue(sent!!.html.contains("Payment received"))
        assertTrue(sent!!.html.contains("0 of 1 purchased photos are available"))
        assertTrue(sent!!.html.contains("125.00"))
        assertFalse(sent!!.html.contains("Get your photo"))
        assertFalse(sent!!.html.contains("All yours."))
    }
    @Test fun `order availability explains review and removal while retaining every financial line`() {
        val order = order()
        val available = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(orderService.getDetail(buyer.id, order.id))["photos"].single()
        assertEquals("AVAILABLE", available["availability"].asText())
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        val review = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(orderService.getDetail(buyer.id, order.id)).get("photos").single()
        assertEquals("REMOVAL_REVIEW", review.get("availability")?.asText())
        assertNotNull(review.get("downloadUrl"))
        authorize()
        val removed = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(orderService.getDetail(buyer.id, order.id)).get("photos").single()
        assertEquals("REMOVED", removed.get("availability")?.asText())
        assertEquals("PHOTO_REMOVED", removed.get("unavailableReason")?.asText())
        assertEquals(photo.id.toString(), removed.get("id").asText())
    }
    // Diagnostic qualification of the existing request-lock/FK contract, not a feature RED.
    @Test fun `request lock and settlement notice foreign keys do not invert photo locking`() {
        val order = order(guest = true)
        authorize()
        val requestLocked = java.util.concurrent.CountDownLatch(1)
        val photoLocked = java.util.concurrent.CountDownLatch(1)
        val pool = java.util.concurrent.Executors.newFixedThreadPool(2)
        try {
            val authorization = pool.submit {
                tx.executeWithoutResult {
                    requests.findForUpdate(item.requestId)
                    requestLocked.countDown()
                    check(photoLocked.await(10, java.util.concurrent.TimeUnit.SECONDS))
                    photos.findAllByIdForUpdate(listOf(photo.id))
                }
            }
            val settlement = pool.submit {
                check(requestLocked.await(10, java.util.concurrent.TimeUnit.SECONDS))
                tx.executeWithoutResult {
                    photos.findAllByIdForUpdate(listOf(photo.id))
                    photoLocked.countDown()
                    removalNotices.onOrderSettled(order.id, Instant.now())
                }
            }
            settlement.get(30, java.util.concurrent.TimeUnit.SECONDS)
            authorization.get(30, java.util.concurrent.TimeUnit.SECONDS)
            assertEquals(1, jdbc.queryForObject("SELECT COUNT(*) FROM photo_removal_history WHERE metadata->>'orderId' = ?", Int::class.java, order.id.toString()))
        } finally { pool.shutdownNow() }
    }
    // Supplemental regressions for the delivery/notice behavior already exercised in RED.
    @Test fun `removal preserves paid rows and seller history while hiding seller and ledger images`() {
        val order = order(pending = true)
        val ref = "pi_${UUID.randomUUID()}"
        payments.saveAndFlush(Payment(orderId = order.id, provider = "paymongo", providerRef = ref, amountPhp = BigDecimal("125.00")))
        webhook.settlePaymentIntent(ref, "pay_synthetic")
        awaitReceiptDelivery(order.id)
        // Existing refund representation: negative ledger row linked to its sale.
        val sale = transactions.findByOrderIdAndPhotoIdAndIsRefund(order.id, photo.id, false)!!
        transactions.saveAndFlush(Transaction(paidAt = sale.paidAt, photographerId = owner.id, eventId = event.id,
            photoId = photo.id, orderId = order.id, buyerId = buyer.id, amountKeptPhp = sale.amountKeptPhp.negate(),
            isRefund = true, refundOf = sale.id))
        orders.findById(order.id).orElseThrow().also { it.status = OrderStatus.REFUNDED; orders.saveAndFlush(it) }
        fun financialRows() = listOf("orders", "order_items", "payments", "download_grants", "transactions").map { table ->
            jdbc.queryForList("SELECT * FROM $table WHERE ${if (table == "orders") "id" else "order_id"} = ? ORDER BY ${if (table in listOf("order_items", "download_grants")) "photo_id" else "id"}", order.id)
        }
        val before = financialRows()
        fun saleRefundTotals() = jdbc.queryForList("SELECT is_refund, SUM(amount_kept_php) AS total FROM transactions WHERE order_id = ? GROUP BY is_refund ORDER BY is_refund", order.id)
        val totalsBefore = saleRefundTotals()
        assertEquals(listOf(BigDecimal("93.75"), BigDecimal("-93.75")), totalsBefore.map { it["total"] })
        val params = com.quickpitik.common.PaginationParams(0, 20)
        assertNotNull(purchases.listPhotoPurchases(owner.id, photo.id, order.id, params).photo.thumbnailUrl)
        assertEquals(2, ledgerReader.list(owner.id, params).items.size)
        assertTrue(ledgerReader.list(owner.id, params).items.all { it.thumbnailUrl != null })
        authorize()
        tx.executeWithoutResult { removalNotices.onRemovalAuthorized(photo.id, item.requestId, Instant.now()) }
        assertEquals(before, financialRows())
        assertEquals(totalsBefore, saleRefundTotals())
        val purchase = purchases.listPhotoPurchases(owner.id, photo.id, order.id, params)
        assertNull(purchase.photo.thumbnailUrl)
        assertEquals(order.id, purchase.selectedPurchase?.orderId)
        assertTrue(ledgerReader.list(owner.id, params).items.all { it.thumbnailUrl == null })
        val read = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(messageReader.list(buyer.id, params).single())
        assertEquals("photo_removed", read["kind"].asText())
        assertEquals(item.requestId.toString(), read["photoRemovalRequestId"].asText())
    }
    @Test fun `guest capability cannot reveal or deliver another removed order`() {
        val first = order(guest = true)
        val other = order(guest = true)
        authorize()
        val foreign = tokens.issue(first, OrderCapability.BUNDLE)
        assertEquals(com.quickpitik.common.ErrorCodes.ORDER_NOT_FOUND, assertFailsWith<com.quickpitik.exception.NotFoundException> {
            bundles.prepare(other.id, foreign)
        }.code)
        assertEquals(HttpStatus.GONE, assertFailsWith<ApiException> {
            bundles.prepare(first.id, foreign)
        }.status)
    }
    @Test fun `reviewKeepsPaidAccessButAuthorizationStopsIt`() {
        val order = order()
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        val review = download(order)
        assertEquals(200, review.statusCode())
        assertContentEquals(bytes, review.body())
        val itemIds = orderItems.findByIdOrderId(order.id).map { it.id }
        val grantRows = jdbc.queryForList("SELECT * FROM download_grants WHERE order_id = ?", order.id)
        authorize()
        val detail = orderService.getDetail(buyer.id, order.id).photos.single()
        assertAll(
            { assertEquals(410, download(order).statusCode()) },
            { assertNull(detail.downloadUrl) }, { assertNull(detail.previewUrl) }, { assertNull(detail.thumbnailUrl) },
            { assertEquals(itemIds, orderItems.findByIdOrderId(order.id).map { it.id }) },
            { assertEquals(grantRows, jdbc.queryForList("SELECT * FROM download_grants WHERE order_id = ?", order.id)) },
        )
    }
    @Test fun `oldBundleTokenCannotBypassRemoval`() {
        val order = order(guest = true)
        val token = tokens.issue(order, OrderCapability.BUNDLE)
        val prepared = bundles.prepare(order.id, token)
        authorize()
        val output = ByteArrayOutputStream()
        assertEquals(HttpStatus.GONE, assertFailsWith<ApiException> { bundles.writeTo(prepared, output) }.status)
        assertEquals(0, output.size())
    }
    @Test fun `latePaymentKeepsLedgerWithoutReenablingDownload`() {
        val order = order(pending = true)
        val ref = "pi_${UUID.randomUUID()}"
        val payment = payments.saveAndFlush(Payment(orderId = order.id, provider = "paymongo", providerRef = ref, amountPhp = BigDecimal("125.00")))
        authorize()
        webhook.settlePaymentIntent(ref, "pay_${UUID.randomUUID()}")
        assertEquals(1, inbox.findAll().count { it.orderId == order.id && it.kindWire == "photo_removed" })
        val ledger = jdbc.queryForList("SELECT * FROM transactions WHERE order_id = ?", order.id)
        assertAll(
            { assertEquals(OrderStatus.FULFILLED, orders.findById(order.id).orElseThrow().status) },
            { assertEquals(PaymentStatus.SUCCEEDED, payments.findById(payment.id).orElseThrow().status) },
            { assertEquals(BigDecimal("125.00"), payments.findById(payment.id).orElseThrow().amountPhp) },
            { assertEquals(1, ledger.size) }, { assertEquals(BigDecimal("93.75"), ledger.single()["amount_kept_php"]) },
            { assertEquals(1, orderItems.findByIdOrderId(order.id).size) },
            { assertEquals(410, download(order).statusCode()) },
            { assertNull(orderService.getDetail(buyer.id, order.id).photos.single().downloadUrl) },
        )
    }
    @Test fun `ownerDownloadAlsoStops`() {
        tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
        assertEquals(photo.id, metadata.getOwned(owner.id, photo.id).photoId)
        assertTrue(ownerBundles.prepareDownloadUrl(owner.id, event.id).url.isNotBlank())
        authorize()
        assertAll(
            { assertEquals(HttpStatus.GONE, assertFailsWith<ApiException> { ownerEvents.getDownload(owner.id, photo.id) }.status) },
            { assertEquals(HttpStatus.GONE, assertFailsWith<ApiException> { ownerBundles.prepare(owner.id, event.id) }.status) },
            { assertNull(photoMapper.toLibraryDto(photos.findById(photo.id).orElseThrow(), 1,
                photo.id in photos.findDeliveryAvailableIds(owner.id, listOf(photo.id))).thumbnailUrl) },
            { assertEquals("NOT_FOUND", assertFailsWith<com.quickpitik.exception.NotFoundException> { metadata.getOwned(owner.id, photo.id) }.code) },
        )
    }
}
