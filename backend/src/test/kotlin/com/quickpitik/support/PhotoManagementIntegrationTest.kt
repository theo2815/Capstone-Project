package com.quickpitik.support

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProperties
import com.quickpitik.entity.*
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.*
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.cart.CartService
import com.quickpitik.service.photographer.PhotoIndexingService
import com.quickpitik.service.photographer.PhotographerPhotoService
import com.quickpitik.service.photos.PhotoService
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.TransactionDefinition
import org.springframework.transaction.support.TransactionTemplate
import java.io.IOException
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.*

// Real PostgreSQL regression checks for photographer trash and permanent deletion.
// Fault injection forces the interleavings that ordinary mock-only tests miss.
class PhotoManagementIntegrationTest : PostgresIntegrationTest() {
    private fun <T> anyArg(): T = Mockito.any()
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var coverage: EventPhotographerRepository
    @Autowired lateinit var items: OrderItemRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var cart: CartService
    @Autowired lateinit var publicPhotos: PhotoService
    @Autowired lateinit var manager: PhotographerPhotoService
    @Autowired lateinit var uploadIntents: PhotoUploadIntentRepository
    @Autowired lateinit var grants: DownloadGrantRepository
    @Autowired lateinit var reconciler: com.quickpitik.service.orders.PaymongoCheckoutReconciler
    @Autowired lateinit var tx: TransactionTemplate
    lateinit var event: Event
    lateinit var owner: User
    lateinit var photo: Photo

    @BeforeEach fun seed() {
        owner = users.save(User(email = "review-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Review", role = Role.PHOTOGRAPHER))
        event = events.save(Event(slug = "review-${UUID.randomUUID()}", name = "Review", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE, photoCount = 1))
        photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "review/${UUID.randomUUID()}/original.jpg", watermarkS3Key = "review/${UUID.randomUUID()}/watermark.jpg", pricePhp = BigDecimal("125")))
        coverage.saveAndFlush(EventPhotographer(id = EventPhotographerId(event.id, owner.id), photoCount = 1, firstUploadAt = OffsetDateTime.now()))
    }

    private fun deletingWith(storage: StorageService, coverageRepo: EventPhotographerRepository = coverage) =
        PhotographerPhotoService(photos, uploadIntents, coverageRepo, items, storage, events, grants, reconciler, tx)

    @Test fun `trash is excluded from real gallery queries while owner trash includes it`() {
        manager.trash(owner.id, event.id, listOf(photo.id))
        val page = PageRequest.of(0, 20)
        assertEquals(0L, photos.findForEventNoBib(event.id, OffsetDateTime.now().plusSeconds(2), 1, page).totalElements)
        assertEquals(0L, photos.searchForEvent(event.id, PhotoStatus.LIVE, "", page).totalElements)
        assertEquals(0L, photos.findPhotographerLibraryByStatus(event.id, owner.id, PhotoStatus.LIVE, page).totalElements)
        assertEquals(1L, photos.findPhotographerTrash(event.id, owner.id, page).totalElements)
    }

    @Test fun `trashed free photo must not resolve an unauthenticated original download`() {
        event.pricingMode = EventPricingMode.FREE
        events.saveAndFlush(event)
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertFailsWith<NotFoundException> { publicPhotos.freeDownload(event.id, photo.id) }
    }

    @Test fun `cart must stop returning a trashed photo and fresh preview URL`() {
        val runner = users.save(User(email = "runner-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Runner", role = Role.RUNNER))
        cart.add(runner.id, photo.id, event.id)
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertTrue(cart.list(runner.id).isEmpty(), "Trashed item is still returned with a newly minted preview URL")
    }

    @Test fun `expired unpaid order must not be reported as a sale`() {
        val order = orders.saveAndFlush(Order(eventId = event.id, recipientEmail = "runner@test.local", paymentMethodWire = "gcash", status = OrderStatus.EXPIRED, totalPhp = photo.pricePhp))
        items.saveAndFlush(OrderItem(id = OrderItemId(order.id, photo.id), pricePhpAtPurchase = photo.pricePhp))
        manager.trash(owner.id, event.id, listOf(photo.id))
        val result = manager.permanentDelete(owner.id, event.id, listOf(photo.id))
        assertTrue(result.blockedSold.isEmpty(), "No payment or grant exists, but the expired order blocks deletion as already sold")
    }

    @Test fun `restore must not succeed after permanent delete has destroyed the original but failed on derivative`() {
        manager.trash(owner.id, event.id, listOf(photo.id))
        val keys = mutableSetOf(photo.s3Key, photo.watermarkS3Key!!)
        val storage = Mockito.mock(StorageService::class.java)
        Mockito.doAnswer { call ->
            val key = call.getArgument<String>(0)
            if (key == photo.watermarkS3Key) throw IOException("Injected derivative delete outage")
            keys.remove(key)
            null
        }.`when`(storage).delete(Mockito.anyString())
        val deleting = deletingWith(storage)
        deleting.permanentDelete(owner.id, event.id, listOf(photo.id))
        val restored = manager.restore(owner.id, event.id, listOf(photo.id))
        assertTrue(restored == 0 || photo.s3Key in keys, "Restore reported success for a photo whose original has already been destroyed")
    }

    @Test fun `an irreversible deletion claim prevents a concurrent restore`() {
        manager.trash(owner.id, event.id, listOf(photo.id))
        val storage = Mockito.mock(StorageService::class.java)
        var restored = false
        Mockito.doAnswer {
            if (!restored) {
                assertEquals(0, manager.restore(owner.id, event.id, listOf(photo.id)))
                restored = true
            }
            null
        }.`when`(storage).delete(Mockito.anyString())
        val deleting = deletingWith(storage)
        deleting.permanentDelete(owner.id, event.id, listOf(photo.id))
        assertFalse(photos.existsById(photo.id))
    }

    @Test fun `permanent delete keeps committed upload receipt terminal and restarts cleanup`() {
        val now = OffsetDateTime.now()
        val receipt = uploadIntents.saveAndFlush(
            PhotoUploadIntent(
                photographerId = owner.id,
                eventId = event.id,
                contentHash = "a".repeat(64),
                contentType = "image/jpeg",
                sizeBytes = 1,
                stagingKey = "staging/photo-uploads/${owner.id}/${event.id}/${photo.id}/original",
                originalKey = photo.s3Key,
                state = PhotoUploadIntentState.COMMITTED,
                resultPhotoId = photo.id,
                putExpiresAt = now.plusMinutes(15),
                expiresAt = now.plusHours(24),
                cleanupAfter = null,
                cleanedAt = now.minusDays(8),
            ),
        )
        manager.trash(owner.id, event.id, listOf(photo.id))

        assertEquals(listOf(photo.id), manager.permanentDelete(owner.id, event.id, listOf(photo.id)).deleted)

        val stored = uploadIntents.findById(receipt.photoId).orElseThrow()
        assertEquals(PhotoUploadIntentState.COMMITTED, stored.state)
        assertNull(stored.resultPhotoId)
        assertNotNull(stored.cleanupAfter)
        assertNull(stored.cleanedAt)
    }

    @Test fun `trash counter update must preserve a concurrent upload increment`() {
        val key = EventPhotographerId(event.id, owner.id)
        val wrappedCoverage = Mockito.mock(EventPhotographerRepository::class.java)
        val separateTx = TransactionTemplate(tx.transactionManager!!).apply { propagationBehavior = TransactionDefinition.PROPAGATION_REQUIRES_NEW }
        Mockito.`when`(wrappedCoverage.findById(key)).thenAnswer {
            val snapshot = coverage.findById(key)
            separateTx.execute { coverage.upsertOnUpload(event.id, owner.id, OffsetDateTime.now()) }
            snapshot
        }
        Mockito.`when`(wrappedCoverage.adjustPhotoCount(event.id, owner.id, -1)).thenAnswer {
            separateTx.execute { coverage.upsertOnUpload(event.id, owner.id, OffsetDateTime.now()) }
            coverage.adjustPhotoCount(event.id, owner.id, -1)
        }
        val deleting = deletingWith(Mockito.mock(StorageService::class.java), wrappedCoverage)
        tx.execute { deleting.trash(owner.id, event.id, listOf(photo.id)) }
        assertEquals(1, coverage.findById(key).orElseThrow().photoCount, "Initial 1 + concurrent upload 1 - trash 1 must equal 1")
    }

    @Test fun `trash must also update the event photo count used by public event DTOs`() {
        event.photoCount = 1
        events.saveAndFlush(event)
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertEquals(0, events.findById(event.id).orElseThrow().photoCount, "Public event photoCount still includes the trashed photo")
    }

    @Test fun `background indexing must not clear a committed trash timestamp`() {
        val wrappedPhotos = Mockito.mock(PhotoRepository::class.java)
        Mockito.`when`(wrappedPhotos.findById(photo.id)).thenAnswer {
            val snapshot = photos.findById(photo.id)
            manager.trash(owner.id, event.id, listOf(photo.id))
            snapshot
        }
        Mockito.`when`(wrappedPhotos.save(anyArg<Photo>())).thenAnswer { photos.save(it.getArgument<Photo>(0)) }
        val indexer = PhotoIndexingService(wrappedPhotos, Mockito.mock(StorageService::class.java), Mockito.mock(FaceBibProvider::class.java), AiApiProperties(enabled = false), AiProperties(), Mockito.mock(ApplicationEventPublisher::class.java), tx, SimpleMeterRegistry(), Mockito.mock(com.quickpitik.service.photos.PhotoWorkFence::class.java), Mockito.mock(com.quickpitik.service.photos.PhotoFaceErasureService::class.java))
        Mockito.`when`(wrappedPhotos.skipIndexing(photo.id)).thenAnswer { photos.skipIndexing(photo.id) }
        indexer.index(photo.id)
        assertNotNull(photos.findById(photo.id).orElseThrow().deletedAt, "Background full-entity save resurrected the trashed photo")
    }

    @Test fun `trash and restore are idempotent and preserve lifecycle and counters`() {
        photo.status = PhotoStatus.PROCESSING
        photos.saveAndFlush(photo)
        assertEquals(1, manager.trash(owner.id, event.id, listOf(photo.id, photo.id)))
        assertEquals(0, manager.trash(owner.id, event.id, listOf(photo.id)))
        assertEquals(0, coverage.findById(EventPhotographerId(event.id, owner.id)).orElseThrow().photoCount)
        assertEquals(1, manager.restore(owner.id, event.id, listOf(photo.id, photo.id)))
        assertEquals(0, manager.restore(owner.id, event.id, listOf(photo.id)))
        assertEquals(PhotoStatus.PROCESSING, photos.findById(photo.id).orElseThrow().status)
        assertEquals(1, events.findById(event.id).orElseThrow().photoCount)
        assertEquals(1, coverage.findById(EventPhotographerId(event.id, owner.id)).orElseThrow().photoCount)
    }

    @Test fun `foreign owner and wrong event IDs cannot mutate or purge photos`() {
        assertEquals(0, manager.trash(UUID.randomUUID(), event.id, listOf(photo.id)))
        assertEquals(0, manager.trash(owner.id, UUID.randomUUID(), listOf(photo.id)))
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertEquals(0, manager.restore(UUID.randomUUID(), event.id, listOf(photo.id)))
        val storage = Mockito.mock(StorageService::class.java)
        assertTrue(deletingWith(storage).permanentDelete(UUID.randomUUID(), event.id, listOf(photo.id)).deleted.isEmpty())
        Mockito.verifyNoInteractions(storage)
        assertTrue(photos.existsById(photo.id))
    }

    @Test fun `failed cleanup remains irreversible and a retry removes its objects and row`() {
        manager.trash(owner.id, event.id, listOf(photo.id))
        val storage = Mockito.mock(StorageService::class.java)
        Mockito.doAnswer { throw IOException("R2 temporarily unavailable") }.`when`(storage).delete(photo.watermarkS3Key!!)
        val deleting = deletingWith(storage)
        assertEquals(listOf(photo.id), deleting.permanentDelete(owner.id, event.id, listOf(photo.id)).failed)
        assertNotNull(photos.findById(photo.id).orElseThrow().deletionStartedAt)
        assertEquals(0, manager.restore(owner.id, event.id, listOf(photo.id)))
        Mockito.doNothing().`when`(storage).delete(photo.watermarkS3Key!!)
        assertEquals(listOf(photo.id), deleting.permanentDelete(owner.id, event.id, listOf(photo.id)).deleted)
        assertFalse(photos.existsById(photo.id))
    }

    @Test fun `a stale full entity save cannot clear trash or its irreversible claim`() {
        val stale = photos.findById(photo.id).orElseThrow()
        manager.trash(owner.id, event.id, listOf(photo.id))
        tx.executeWithoutResult { photos.startDeletion(photo.id, OffsetDateTime.now()) }
        stale.indexingStatus = IndexingStatus.SKIPPED
        photos.saveAndFlush(stale)
        val fresh = photos.findById(photo.id).orElseThrow()
        assertNotNull(fresh.deletedAt)
        assertNotNull(fresh.deletionStartedAt)
    }

    @Test fun `background indexing update cannot clobber a concurrent watermark publish`() {
        photo.status = PhotoStatus.PROCESSING
        photo.watermarkS3Key = null
        photo.thumbnailS3Key = null
        photo.publishedAt = null
        photos.saveAndFlush(photo)
        val separateTx = TransactionTemplate(tx.transactionManager!!).apply {
            propagationBehavior = TransactionDefinition.PROPAGATION_REQUIRES_NEW
        }

        tx.executeWithoutResult {
            val stale = photos.findById(photo.id).orElseThrow()
            separateTx.execute {
                assertEquals(1, photos.publishWatermarked(photo.id, "review/watermark.jpg", 1L, 2L, 3L))
            }
            stale.indexingStatus = IndexingStatus.SKIPPED
            photos.saveAndFlush(stale)
        }

        val fresh = photos.findById(photo.id).orElseThrow()
        assertEquals(IndexingStatus.SKIPPED, fresh.indexingStatus)
        assertEquals(PhotoStatus.LIVE, fresh.status)
        assertEquals("review/watermark.jpg", fresh.watermarkS3Key)
    }

    @Test fun `publishWatermarked only flips an eligible processing photo`() {
        val processing = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "review/processing.jpg", pricePhp = BigDecimal("125"), status = PhotoStatus.PROCESSING))
        val hidden = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "review/hidden.jpg", pricePhp = BigDecimal("125"), status = PhotoStatus.HIDDEN))
        val trashed = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "review/trashed.jpg", pricePhp = BigDecimal("125"), status = PhotoStatus.PROCESSING, deletedAt = OffsetDateTime.now()))

        assertEquals(1, inTx { photos.publishWatermarked(processing.id, "review/watermark.jpg", 1L, 2L, 3L) })
        assertEquals(0, inTx { photos.publishWatermarked(photo.id, "review/watermark.jpg", 1L, 2L, 3L) })
        assertEquals(0, inTx { photos.publishWatermarked(hidden.id, "review/watermark.jpg", 1L, 2L, 3L) })
        assertEquals(0, inTx { photos.publishWatermarked(trashed.id, "review/watermark.jpg", 1L, 2L, 3L) })
        assertEquals(PhotoStatus.LIVE, photos.findById(processing.id).orElseThrow().status)
    }

    @Test fun `purchased photos and existing grants survive permanent deletion requests`() {
        val order = orders.saveAndFlush(Order(eventId = event.id, recipientEmail = "buyer@test.local", paymentMethodWire = "gcash", status = OrderStatus.FULFILLED, totalPhp = photo.pricePhp))
        items.saveAndFlush(OrderItem(id = OrderItemId(order.id, photo.id), pricePhpAtPurchase = photo.pricePhp))
        grants.saveAndFlush(DownloadGrant(id = DownloadGrantId(order.id, photo.id), grantedUntil = OffsetDateTime.now().plusDays(7)))
        manager.trash(owner.id, event.id, listOf(photo.id))
        val storage = Mockito.mock(StorageService::class.java)
        val result = deletingWith(storage).permanentDelete(owner.id, event.id, listOf(photo.id))
        assertEquals(listOf(photo.id), result.blockedSold)
        assertTrue(grants.existsByIdPhotoId(photo.id))
        assertTrue(photos.existsById(photo.id))
        Mockito.verifyNoInteractions(storage)
    }

    @Test fun `pending checkouts are protected separately from sales`() {
        val order = orders.saveAndFlush(Order(eventId = event.id, recipientEmail = "runner@test.local", paymentMethodWire = "gcash", status = OrderStatus.PENDING, totalPhp = photo.pricePhp))
        items.saveAndFlush(OrderItem(id = OrderItemId(order.id, photo.id), pricePhpAtPurchase = photo.pricePhp))
        manager.trash(owner.id, event.id, listOf(photo.id))
        val storage = Mockito.mock(StorageService::class.java)
        val result = deletingWith(storage).permanentDelete(owner.id, event.id, listOf(photo.id))
        assertTrue(result.blockedSold.isEmpty())
        assertEquals(listOf(photo.id), result.blockedPending)
        Mockito.verifyNoInteractions(storage)
    }

    // Face search reads through findByEventAndPersonIds, which is not one of the
    // gallery queries the first test covers; its deleted_at filter lives only in
    // JPQL (2026-09-11 test-suite audit).
    @Test fun `face search never returns a trashed photo`() {
        val personId = "person-${UUID.randomUUID()}"
        photo.facePersons.add(PhotoFacePersonEmbed(faceIndex = 0, aiPersonId = personId))
        photos.saveAndFlush(photo)
        val page = PageRequest.of(0, 20)
        assertEquals(1L, photos.findByEventAndPersonIds(event.id, PhotoStatus.LIVE, listOf(personId), page).totalElements)
        manager.trash(owner.id, event.id, listOf(photo.id))
        assertEquals(0L, photos.findByEventAndPersonIds(event.id, PhotoStatus.LIVE, listOf(personId), page).totalElements)
    }

    // PhotoDto.cleanUrl is handed out for exactly the ids this query returns,
    // and the grant's expiry is enforced only by its `grantedUntil > CURRENT_TIMESTAMP`.
    @Test fun `an expired or foreign download grant hands out no clean url`() {
        val buyer = users.save(User(email = "buyer-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Buyer", role = Role.RUNNER))
        val stranger = users.save(User(email = "stranger-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Stranger", role = Role.RUNNER))
        val order = orders.saveAndFlush(Order(userId = buyer.id, eventId = event.id, recipientEmail = buyer.email, paymentMethodWire = "gcash", status = OrderStatus.FULFILLED, totalPhp = photo.pricePhp))
        val grant = grants.saveAndFlush(DownloadGrant(id = DownloadGrantId(order.id, photo.id), grantedUntil = OffsetDateTime.now().plusDays(1)))
        assertEquals(listOf(photo.id), grants.findOwnedPhotoIdsByUserAndPhotoIds(buyer.id, listOf(photo.id)))
        assertTrue(grants.findOwnedPhotoIdsByUserAndPhotoIds(stranger.id, listOf(photo.id)).isEmpty())
        grant.grantedUntil = OffsetDateTime.now().minusMinutes(1)
        grants.saveAndFlush(grant)
        assertTrue(grants.findOwnedPhotoIdsByUserAndPhotoIds(buyer.id, listOf(photo.id)).isEmpty())
    }

    // Each conditional update gets its own transaction, as AdminFlagService's
    // does: a @Modifying query without one throws, and reading inside the
    // writing transaction answers from the persistence context, not Postgres.
    private fun inTx(block: () -> Int): Int = tx.execute { block() }!!

    // The flag queue's two conditional updates (P1.5). Mocks never parse JPQL,
    // and a bulk SET binding an @Enumerated(STRING) column is exactly the shape
    // that compiles clean and writes an ordinal at runtime.
    @Test fun `flag hide and restore move a watermarked photo between HIDDEN and LIVE`() {
        assertEquals(1, inTx { photos.hideForFlag(photo.id) })
        assertEquals(PhotoStatus.HIDDEN, photos.findById(photo.id).get().status)
        assertEquals(0, inTx { photos.hideForFlag(photo.id) }, "hiding an already-hidden photo must be a no-op")
        assertEquals(1, inTx { photos.unhideForFlag(photo.id, PhotoStatus.LIVE) })
        assertEquals(PhotoStatus.LIVE, photos.findById(photo.id).get().status)
        assertEquals(0, inTx { photos.unhideForFlag(photo.id, PhotoStatus.LIVE) }, "restore must require HIDDEN")
    }

    // The invariant P1.5 exists for: a photo hidden before its watermark
    // rendered has no derived asset, so LIVE would serve the clean original as
    // the public thumbnail. It goes back to the sweep instead.
    @Test fun `a photo hidden mid-pipeline restores to PROCESSING and re-enters the watermark backlog`() {
        val pending = photos.saveAndFlush(
            Photo(
                eventId = event.id,
                photographerId = owner.id,
                s3Key = "review/${UUID.randomUUID()}/original.jpg",
                pricePhp = BigDecimal("125"),
                status = PhotoStatus.PROCESSING,
                uploadedAt = OffsetDateTime.now().minusMinutes(10),
                processingAttempts = 3,
            ),
        )
        assertEquals(1, inTx { photos.hideForFlag(pending.id) }, "a PROCESSING photo must still be hideable")
        assertEquals(1, inTx { photos.unhideForFlag(pending.id, PhotoStatus.PROCESSING) })
        val restored = photos.findById(pending.id).get()
        assertEquals(PhotoStatus.PROCESSING, restored.status)
        assertEquals(0, restored.processingAttempts, "an exhausted budget would strand the row outside the sweep")
        val backlog = photos.findWatermarkBacklog(3, OffsetDateTime.now(), PageRequest.of(0, 200)).map { it.id }
        assertTrue(pending.id in backlog, "the sweep is what re-renders the preview after a restore")
    }
}
