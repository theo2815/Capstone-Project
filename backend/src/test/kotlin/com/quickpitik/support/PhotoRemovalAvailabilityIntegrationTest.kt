package com.quickpitik.support

import com.quickpitik.dto.orders.CreateOrderItem
import com.quickpitik.dto.orders.CreateOrderRequest
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponse
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseEnvelope
import com.quickpitik.dto.orders.PaymongoCheckoutSessionResponseAttributes
import com.quickpitik.entity.*
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.*
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.orders.*
import com.quickpitik.service.photographer.PhotoMetadataService
import com.quickpitik.service.photographer.PublicPhotographerService
import com.quickpitik.service.cart.CartService
import com.quickpitik.service.photos.PhotoAvailabilityService
import com.quickpitik.service.photos.PhotoService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertAll
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.data.domain.PageRequest
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.io.ByteArrayInputStream
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import kotlin.test.*

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PhotoRemovalAvailabilityIntegrationTest : PostgresIntegrationTest() {
    @LocalServerPort var port: Int = 0
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var orderItems: OrderItemRepository
    @Autowired lateinit var grants: DownloadGrantRepository
    @Autowired lateinit var availability: PhotoAvailabilityService
    @Autowired lateinit var photoService: PhotoService
    @Autowired lateinit var metadata: PhotoMetadataService
    @Autowired lateinit var photographerSettings: PhotographerSettingsRepository
    @Autowired lateinit var profiles: PublicPhotographerService
    @Autowired lateinit var carts: CartService
    @Autowired lateinit var mapper: EventDtoMapper
    @Autowired lateinit var checkout: OrderService
    @Autowired lateinit var tokens: OrderAccessTokenService
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var jdbc: JdbcTemplate
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var provider: AiApiClient
    @MockitoBean lateinit var paymongo: PaymongoClient
    private lateinit var event: Event
    private lateinit var owner: User
    private lateinit var photo: Photo
    private lateinit var item: PhotoRemovalItem
    private val page = PageRequest.of(0, 20)

    @BeforeEach fun fixture() {
        owner = users.saveAndFlush(User(email = "availability-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Photographer", role = Role.PHOTOGRAPHER))
        photographerSettings.saveAndFlush(PhotographerSettings(userId = owner.id, verificationStatus = VerificationStatus.APPROVED))
        event = events.saveAndFlush(Event(slug = "availability-${UUID.randomUUID()}", name = "Cebu run",
            date = LocalDate.of(2026, 10, 5), location = "Cebu", status = EventStatus.ACTIVE, photoCount = 1))
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
            s3Key = "synthetic/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("125"), phash = UUID.randomUUID().leastSignificantBits,
            uploadedAt = OffsetDateTime.now().minusMinutes(5)).apply {
            bibs.add(PhotoBibEmbed(bibNumber = "123", ocrConfidence = BigDecimal.ONE))
            facePersons.add(PhotoFacePersonEmbed(faceIndex = 0, aiPersonId = "person-${id}"))
        })
        item = verifiedItem()
        Mockito.`when`(storage.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://synthetic.test/photo")
        Mockito.`when`(paymongo.createCheckoutSession(anyArg(), anyArg())).thenReturn(
            PaymongoCheckoutSessionResponse(PaymongoCheckoutSessionResponseEnvelope(id = "cs_${UUID.randomUUID()}",
                attributes = PaymongoCheckoutSessionResponseAttributes(checkoutUrl = "https://pay.test/checkout"))))
    }

    private fun verifiedItem(): PhotoRemovalItem {
        val request = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id, eventIdSnapshot = event.id,
            eventNameSnapshot = event.name, requesterContact = "Support contact", supportReference = "Support-${UUID.randomUUID()}",
            requesterReason = "Privacy", publicToOwnerReason = "Verified privacy request", creatorId = UUID.randomUUID(),
            creationCommandId = UUID.randomUUID(), status = RemovalRequestStatus.OPEN))
        return items.saveAndFlush(PhotoRemovalItem(requestId = request.id, photoIdSnapshot = photo.id,
            photoId = photo.id, ownerIdSnapshot = owner.id, state = RemovalItemState.VERIFIED,
            matchSource = RemovalMatchSource.MANUAL))
    }

    private fun <T> anyArg(): T = Mockito.any()

    private fun restrict() = tx.executeWithoutResult { availability.refreshRemovalState(setOf(photo.id)) }
    private fun publicGalleryIds(eventId: UUID) = photos.findForEventNoBib(eventId, OffsetDateTime.now(), 1, page).content.map { it.id }
    private fun publicFaceMatchIds(eventId: UUID) = photos.findByEventAndPersonIds(eventId, PhotoStatus.LIVE,
        listOf("person-${photo.id}"), page).content.map { it.id }
    private data class Download(val status: Int)
    private fun existingPaidBuyerDownloadDuringReview(order: Order): Download {
        val token = tokens.issue(order, OrderCapability.BUNDLE)
        val bytes = byteArrayOf(1, 2, 3)
        Mockito.`when`(storage.open(photo.s3Key)).thenAnswer { ByteArrayInputStream(bytes) }
        val request = HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/orders/${order.id}/download-bundle?token=$token&photo=${photo.id}"))
            .timeout(java.time.Duration.ofSeconds(10)).GET().build()
        val response = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofByteArray())
        assertContentEquals(bytes, response.body())
        return Download(response.statusCode())
    }

    @Test fun `reviewBlocksPublicPathsButNotAnExistingGrant`() {
        val order = orders.saveAndFlush(Order(eventId = event.id, recipientEmail = "buyer@test.local",
            paymentMethodWire = "gcash", status = OrderStatus.PAID, totalPhp = BigDecimal("125")))
        orderItems.saveAndFlush(OrderItem(id = OrderItemId(order.id, photo.id), pricePhpAtPurchase = BigDecimal("125")))
        grants.saveAndFlush(DownloadGrant(id = DownloadGrantId(order.id, photo.id), grantedUntil = OffsetDateTime.now().plusDays(1)))
        restrict()
        val reviewPhoto = photos.findById(photo.id).orElseThrow()
        val eventId = event.id
        assertAll(
            { assertFalse(reviewPhoto.isSellable) },
            { assertEquals(emptyList<UUID>(), publicGalleryIds(eventId)) },
            { assertEquals(emptyList<UUID>(), publicFaceMatchIds(eventId)) },
            { assertEquals(200, existingPaidBuyerDownloadDuringReview(order).status) },
            { assertEquals(RemovalState.REVIEW, reviewPhoto.removalState) },
            { assertEquals(PhotoStatus.LIVE, reviewPhoto.status) },
            { assertNull(reviewPhoto.deletedAt) },
            { assertTrue(photos.searchForEvent(eventId, PhotoStatus.LIVE, "123", page).isEmpty) },
            { assertTrue(photos.findPhotographerLibraryByStatus(eventId, owner.id, PhotoStatus.LIVE, page).isEmpty) },
            { assertEquals(listOf(photo.id), photos.findPhotographerLibrary(eventId, owner.id, page).content.map { it.id }) },
            { assertFalse(photos.findNearestByPhash(photo.phash!!).any { it[1] == eventId }) },
            { assertEquals(0, mapper.toListDto(event).photoCount) },
            { assertEquals(0, mapper.toDetailDto(event).photoCount) },
            { assertEquals(1, mapper.toAdminListDto(event).photoCount) },
        )
    }

    @Test fun `directPhotoLookupCannotBypassRestriction`() {
        // A free public event and opted-in metadata would otherwise expose the original and EXIF.
        event.pricingMode = EventPricingMode.FREE
        events.saveAndFlush(event)
        jdbc.update("INSERT INTO event_photographer(event_id, photographer_id, metadata_public) VALUES (?, ?, true)", event.id, owner.id)
        restrict()
        assertAll(
            { assertFailsWith<NotFoundException> { photoService.freeDownload(event.id, photo.id) } },
            { assertFailsWith<NotFoundException> { metadata.getPublic(photo.id) } },
            { assertEquals("none", metadata.metadataAccess(event.id, listOf(photos.findById(photo.id).orElseThrow()), null)[photo.id]) },
        )
    }

    @Test fun `checkoutLosesRaceToVerifiedRestriction`() {
        val restricted = CountDownLatch(1)
        val allowCommit = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(2)
        try {
            val review = executor.submit {
                tx.executeWithoutResult {
                    availability.refreshRemovalState(setOf(photo.id))
                    restricted.countDown()
                    check(allowCommit.await(10, TimeUnit.SECONDS))
                }
            }
            assertTrue(restricted.await(10, TimeUnit.SECONDS))
            val purchase = executor.submit<Throwable?> {
                runCatching { checkout.create(null, CreateOrderRequest(items = listOf(CreateOrderItem(photo.id, event.id)),
                    paymentMethod = "gcash", recipientEmail = "new-buyer@test.local"), UUID.randomUUID().toString()) }.exceptionOrNull()
            }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
            var waitingForPhotoLock = false
            while (!waitingForPhotoLock && System.nanoTime() < deadline) {
                waitingForPhotoLock = jdbc.queryForObject("""
                    SELECT EXISTS (SELECT 1 FROM pg_stat_activity
                    WHERE datname = current_database() AND pid <> pg_backend_pid()
                      AND wait_event_type = 'Lock' AND query ILIKE '%photos%')
                """, Boolean::class.java) == true
                if (!waitingForPhotoLock) Thread.sleep(10)
            }
            assertTrue(waitingForPhotoLock, "Checkout must contend for the restricted photo before review commits")
            allowCommit.countDown()
            review.get(15, TimeUnit.SECONDS)
            val failure = purchase.get(15, TimeUnit.SECONDS)
            assertEquals(0, newlyCreatedOrderItemsForRestrictedPhoto())
            assertIs<ConflictException>(failure)
        } finally {
            allowCommit.countDown()
            executor.shutdownNow()
        }
    }

    private fun newlyCreatedOrderItemsForRestrictedPhoto() = jdbc.queryForObject(
        "SELECT count(*) FROM order_items WHERE photo_id = ?", Int::class.java, photo.id)

    @Test fun `review prevents cart entry and public photographer coverage counts`() {
        val handle = "owner-${UUID.randomUUID().toString().take(8)}"
        photographerSettings.saveAndFlush(PhotographerSettings(userId = owner.id, handle = handle,
            verificationStatus = VerificationStatus.APPROVED))
        jdbc.update("INSERT INTO event_photographer(event_id, photographer_id, photo_count) VALUES (?, ?, 1)", event.id, owner.id)
        val buyer = users.saveAndFlush(User(email = "cart-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Buyer", role = Role.RUNNER))
        restrict()
        assertAll(
            { assertFailsWith<NotFoundException> { carts.add(buyer.id, photo.id, event.id) } },
            { assertTrue(profiles.getProfile(handle).events.isEmpty()) },
        )
    }

    @Test fun `another active case keeps review and authorization cannot be refreshed away`() {
        val other = verifiedItem()
        restrict()
        item.state = RemovalItemState.RETAINED
        items.saveAndFlush(item)
        restrict()
        assertEquals(RemovalState.REVIEW, photos.findById(photo.id).orElseThrow().removalState)
        other.state = RemovalItemState.EXCLUDED
        items.saveAndFlush(other)
        restrict()
        assertEquals(RemovalState.NONE, photos.findById(photo.id).orElseThrow().removalState)
        assertEquals(1, mapper.toListDto(event).photoCount)
        assertEquals(1, mapper.toListDtos(listOf(event)).single().photoCount)
        assertEquals(1L, photos.countPublicForPhotographerByEventIds(listOf(event.id), owner.id).single().photoCount)
        val actor = UUID.randomUUID()
        other.state = RemovalItemState.REMOVAL_AUTHORIZED
        other.adminDecision = RemovalDecision.REMOVE
        other.adminDecisionReason = "Verified privacy request"
        other.adminDecisionActorId = actor
        items.saveAndFlush(other)
        tx.executeWithoutResult { availability.authorizeRemoval(photo.id, other.id, actor) }
        restrict()
        assertEquals(RemovalState.REMOVED, photos.findById(photo.id).orElseThrow().removalState)
    }

    @Test fun `authorization requires the matching remove item and recorded decision actor`() {
        val actor = UUID.randomUUID()
        item.state = RemovalItemState.REMOVAL_AUTHORIZED
        item.adminDecision = RemovalDecision.REMOVE
        item.adminDecisionReason = "Verified privacy request"
        item.adminDecisionActorId = actor
        items.saveAndFlush(item)
        assertAll(
            { assertFailsWith<ConflictException> { tx.executeWithoutResult { availability.authorizeRemoval(photo.id, item.id, UUID.randomUUID()) } } },
            { assertFailsWith<ConflictException> { tx.executeWithoutResult { availability.authorizeRemoval(photo.id, UUID.randomUUID(), actor) } } },
        )
        tx.executeWithoutResult { availability.authorizeRemoval(photo.id, item.id, actor) }
        tx.executeWithoutResult { availability.authorizeRemoval(photo.id, item.id, actor) }
        restrict()
        assertEquals(RemovalState.REMOVED, photos.findById(photo.id).orElseThrow().removalState)
        assertNull(photos.findById(photo.id).orElseThrow().purgedAt)
    }

    @Test fun `a resolved retention exception remains restricted and stale saves cannot release it`() {
        item.state = RemovalItemState.RETAINED_RESTRICTED
        item.retentionReviewAt = java.time.Instant.now().plusSeconds(3600)
        items.saveAndFlush(item)
        requests.findById(item.requestId).orElseThrow().apply {
            status = RemovalRequestStatus.RESOLVED
            requests.saveAndFlush(this)
        }
        restrict()
        photo.altText = "A stale processing save"
        photos.saveAndFlush(photo)
        assertEquals(RemovalState.REVIEW, photos.findById(photo.id).orElseThrow().removalState)
    }
}
