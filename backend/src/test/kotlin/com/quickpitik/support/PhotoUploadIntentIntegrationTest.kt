package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotoUploadIntent
import com.quickpitik.entity.PhotoUploadIntentState
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoUploadIntentRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import jakarta.persistence.EntityManager
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PhotoUploadIntentIntegrationTest : PostgresIntegrationTest() {

    @Autowired lateinit var intents: PhotoUploadIntentRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var settings: PhotographerSettingsRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var transactionTemplate: TransactionTemplate
    @Autowired lateinit var entityManager: EntityManager

    @Test
    fun `owner and event scoped lookup does not reveal a foreign intent`() {
        val owner = photographer()
        val foreign = photographer()
        val event = event()
        val intent = intent(owner.id, event.id)

        assertEquals(intent.photoId, intents.findOwned(intent.photoId, owner.id, event.id)?.photoId)
        assertNull(intents.findOwned(intent.photoId, foreign.id, event.id))
        assertNull(intents.findOwned(intent.photoId, owner.id, UUID.randomUUID()))
        assertEquals(intent.photoId, transactionTemplate.execute {
            intents.findOwnedForUpdate(intent.photoId, owner.id, event.id)?.photoId
        })
    }

    @Test
    fun `one request id cannot reserve two intents for the same owner`() {
        val owner = photographer()
        val requestId = UUID.randomUUID()
        intent(owner.id, event().id, requestId)

        assertFailsWith<DataIntegrityViolationException> {
            transactionTemplate.executeWithoutResult {
                intents.saveAndFlush(newIntent(owner.id, event().id, requestId))
            }
        }
        assertEquals(requestId, intents.findByPhotographerIdAndRequestId(owner.id, requestId)?.requestId)
    }

    @Test
    fun `owner lock makes concurrent quota admission observe the first reservation`() {
        val owner = photographer()
        settings.saveAndFlush(PhotographerSettings(userId = owner.id))
        val event = event()
        val firstHasLock = CountDownLatch(1)
        val releaseFirst = CountDownLatch(1)
        val secondStarted = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(2)

        try {
            val first = executor.submit<Boolean> {
                transactionTemplate.execute {
                    check(settings.findByUserIdForUpdate(owner.id) != null)
                    firstHasLock.countDown()
                    check(releaseFirst.await(5, TimeUnit.SECONDS))
                    val quota = intents.activeQuota(owner.id, NOW)
                    if (quota.slots < 1) {
                        intents.saveAndFlush(newIntent(owner.id, event.id))
                        true
                    } else {
                        false
                    }
                } ?: false
            }
            assertTrue(firstHasLock.await(5, TimeUnit.SECONDS))

            val second = executor.submit<Boolean> {
                transactionTemplate.execute {
                    secondStarted.countDown()
                    check(settings.findByUserIdForUpdate(owner.id) != null)
                    val quota = intents.activeQuota(owner.id, NOW)
                    if (quota.slots < 1) {
                        intents.saveAndFlush(newIntent(owner.id, event.id))
                        true
                    } else {
                        false
                    }
                } ?: false
            }
            assertTrue(secondStarted.await(5, TimeUnit.SECONDS))
            assertFailsWith<TimeoutException> { second.get(200, TimeUnit.MILLISECONDS) }

            releaseFirst.countDown()
            assertTrue(first.get(5, TimeUnit.SECONDS))
            assertEquals(false, second.get(5, TimeUnit.SECONDS))
        } finally {
            releaseFirst.countDown()
            executor.shutdownNow()
        }

        val quota = intents.activeQuota(owner.id, NOW)
        assertEquals(1, quota.slots)
        assertEquals(FILE_SIZE, quota.bytes)
    }

    @Test
    fun `quota counts only reservations that can still receive bytes or are being verified`() {
        val owner = photographer()
        val event = event()
        // Signed PUT still open: counts.
        intents.saveAndFlush(newIntent(owner.id, event.id, createdAt = NOW.minusMinutes(5)))
        // PUT window closed five minutes ago; still committable for a day, but must not hold a slot.
        intents.saveAndFlush(newIntent(owner.id, event.id, createdAt = NOW.minusMinutes(20)))
        // Verification in progress after the PUT window: counts.
        intents.saveAndFlush(newIntent(owner.id, event.id, createdAt = NOW.minusMinutes(30)).apply {
            state = PhotoUploadIntentState.VERIFYING
            leaseToken = UUID.randomUUID()
            leaseUntil = NOW.plusMinutes(2)
        })
        // Stale verification lease after the PUT window: must not count.
        intents.saveAndFlush(newIntent(owner.id, event.id, createdAt = NOW.minusMinutes(40)).apply {
            state = PhotoUploadIntentState.VERIFYING
            leaseToken = UUID.randomUUID()
            leaseUntil = NOW.minusMinutes(1)
        })

        val quota = intents.activeQuota(owner.id, NOW)

        assertEquals(2, quota.slots)
        assertEquals(FILE_SIZE * 2, quota.bytes)
    }

    @Test
    fun `reuse finds only the owner's open request-id-less reservation for identical bytes`() {
        val owner = photographer()
        val other = photographer()
        val event = event()
        val open = intents.saveAndFlush(newIntent(owner.id, event.id, createdAt = NOW.minusMinutes(5)))
        intents.saveAndFlush(newIntent(owner.id, event.id, createdAt = NOW.minusMinutes(13)))   // 2 min left: too short
        intents.saveAndFlush(newIntent(owner.id, event.id, UUID.randomUUID()))                  // website request id
        intents.saveAndFlush(newIntent(other.id, event.id))                                      // another photographer

        val found = intents.findReusablePending(
            owner.id, event.id, "a".repeat(64), "image/jpeg", FILE_SIZE, NOW.plusMinutes(5), NOW,
        )

        assertEquals(open.photoId, found?.photoId)
        assertNull(intents.findReusablePending(
            owner.id, event.id, "b".repeat(64), "image/jpeg", FILE_SIZE, NOW.plusMinutes(5), NOW,
        ))
    }

    @Test
    fun `only the current verification lease can complete an intent`() {
        val owner = photographer()
        val event = event()
        val intent = intent(owner.id, event.id)
        val result = photo(owner.id, event.id)
        val staleToken = UUID.randomUUID()
        val currentToken = UUID.randomUUID()

        assertEquals(1, inTransaction {
            intents.acquireVerification(intent.photoId, staleToken, NOW.plusMinutes(5), NOW)
        })
        assertEquals(0, inTransaction {
            intents.acquireVerification(intent.photoId, currentToken, NOW.plusMinutes(10), NOW.plusMinutes(1))
        })
        assertEquals(1, inTransaction {
            intents.acquireVerification(intent.photoId, currentToken, NOW.plusMinutes(12), NOW.plusMinutes(6))
        })
        assertEquals(0, inTransaction {
            intents.complete(intent.photoId, staleToken, result.id, NOW.plusHours(1), NOW.plusMinutes(7))
        })
        assertEquals(1, inTransaction {
            intents.complete(intent.photoId, currentToken, result.id, NOW.plusHours(1), NOW.plusMinutes(7))
        })

        entityManager.clear()
        val stored = intents.findById(intent.photoId).orElseThrow()
        assertEquals(PhotoUploadIntentState.COMMITTED, stored.state)
        assertEquals(result.id, stored.resultPhotoId)
        assertNull(stored.leaseToken)
        assertNull(stored.leaseUntil)
    }

    @Test
    fun `only the current verifier can release an intent for retry`() {
        val owner = photographer()
        val intent = intent(owner.id, event().id)
        val token = UUID.randomUUID()

        assertEquals(1, inTransaction {
            intents.acquireVerification(intent.photoId, token, NOW.plusMinutes(5), NOW)
        })
        assertEquals(0, inTransaction {
            intents.releaseVerification(intent.photoId, UUID.randomUUID(), NOW.plusMinutes(1))
        })
        assertEquals(1, inTransaction {
            intents.releaseVerification(intent.photoId, token, NOW.plusMinutes(1))
        })
        assertEquals(1, inTransaction {
            intents.acquireVerification(intent.photoId, UUID.randomUUID(), NOW.plusMinutes(6), NOW.plusMinutes(2))
        })
    }

    @Test
    fun `only the current verifier can expire a missing upload after its put deadline`() {
        val owner = photographer()
        val intent = intents.saveAndFlush(
            newIntent(owner.id, event().id).apply {
                // The intent remains recoverable for a day, but this particular
                // signed PUT can no longer create the missing object.
            },
        )
        val token = UUID.randomUUID()
        val afterPutDeadline = intent.putExpiresAt.plusSeconds(1)

        assertEquals(1, inTransaction {
            intents.acquireVerification(intent.photoId, token, afterPutDeadline.plusMinutes(5), afterPutDeadline)
        })
        assertEquals(0, inTransaction {
            intents.expireMissingAfterPut(intent.photoId, UUID.randomUUID(), afterPutDeadline.plusHours(1), afterPutDeadline)
        })
        assertEquals(1, inTransaction {
            intents.expireMissingAfterPut(intent.photoId, token, afterPutDeadline.plusHours(1), afterPutDeadline)
        })

        entityManager.clear()
        assertEquals(PhotoUploadIntentState.EXPIRED, intents.findById(intent.photoId).orElseThrow().state)
    }

    @Test
    fun `deleting a result photo retains the committed receipt`() {
        val owner = photographer()
        val event = event()
        val result = photo(owner.id, event.id)
        val receipt = intents.saveAndFlush(
            newIntent(owner.id, event.id).apply {
                state = PhotoUploadIntentState.COMMITTED
                resultPhotoId = result.id
            },
        )

        transactionTemplate.executeWithoutResult {
            photos.deleteById(result.id)
            photos.flush()
        }
        entityManager.clear()

        val stored = intents.findById(receipt.photoId).orElseThrow()
        assertEquals(PhotoUploadIntentState.COMMITTED, stored.state)
        assertNull(stored.resultPhotoId)
    }

    @Test
    fun `expired and cleanup candidate queries are bounded and state scoped`() {
        val owner = photographer()
        val event = event()
        val expired = intents.saveAndFlush(
            newIntent(owner.id, event.id, expiresAt = NOW.minusDays(1)),
        )
        val rejected = intent(owner.id, event.id)
        val token = UUID.randomUUID()
        assertEquals(1, inTransaction {
            intents.acquireVerification(rejected.photoId, token, NOW.plusMinutes(5), NOW)
        })
        assertEquals(1, inTransaction {
            intents.reject(rejected.photoId, token, "UPLOAD_CHECKSUM_MISMATCH", NOW.minusMinutes(1), NOW)
        })

        assertEquals(
            listOf(expired.photoId),
            intents.findExpiredCandidateIds(NOW, PageRequest.of(0, 1)),
        )
        assertEquals(1, inTransaction {
            intents.expire(expired.photoId, NOW.plusHours(1), NOW)
        })
        assertEquals(
            listOf(rejected.photoId),
            intents.findCleanupCandidateIds(
                NOW,
                listOf(PhotoUploadIntentState.REJECTED),
                PageRequest.of(0, 10),
            ),
        )
    }

    @Test
    fun `cleanup claim is exclusive recovers after a worker crash and retains the receipt`() {
        val owner = photographer()
        val event = event()
        val receipt = intents.saveAndFlush(
            newIntent(owner.id, event.id).apply {
                state = PhotoUploadIntentState.REJECTED
                cleanupAfter = NOW.minusMinutes(1)
            },
        )
        val claimedUntil = NOW.plusMinutes(1)
        val reclaimedUntil = claimedUntil.plusMinutes(1)

        assertEquals(1, inTransaction { intents.claimCleanup(receipt.photoId, NOW, claimedUntil) })
        assertEquals(0, inTransaction { intents.claimCleanup(receipt.photoId, NOW, claimedUntil.plusMinutes(1)) })
        assertTrue(intents.findCleanupCandidateIds(
            NOW,
            listOf(PhotoUploadIntentState.REJECTED),
            PageRequest.of(0, 10),
        ).isEmpty())
        assertEquals(1, inTransaction { intents.claimCleanup(receipt.photoId, claimedUntil, reclaimedUntil) })
        assertEquals(0, inTransaction {
            intents.completeCleanup(receipt.photoId, claimedUntil, NOW, null, NOW)
        })
        assertEquals(1, inTransaction {
            intents.completeCleanup(receipt.photoId, reclaimedUntil, NOW, null, claimedUntil)
        })

        entityManager.clear()
        val stored = intents.findById(receipt.photoId).orElseThrow()
        assertEquals(PhotoUploadIntentState.REJECTED, stored.state)
        assertEquals(NOW, stored.cleanedAt)
        assertEquals(claimedUntil, stored.cleanupCheckedAt)
        assertNull(stored.cleanupAfter)
    }

    @Test
    fun `cleanup retry makes a failed terminal receipt due again without marking it cleaned`() {
        val owner = photographer()
        val receipt = intents.saveAndFlush(
            newIntent(owner.id, event().id).apply {
                state = PhotoUploadIntentState.EXPIRED
                cleanupAfter = NOW.minusMinutes(1)
            },
        )
        val claimedUntil = NOW.plusMinutes(1)
        val retryAt = NOW.plusMinutes(2)

        assertEquals(1, inTransaction { intents.claimCleanup(receipt.photoId, NOW, claimedUntil) })
        assertEquals(1, inTransaction { intents.retryCleanup(receipt.photoId, claimedUntil, retryAt, NOW) })

        entityManager.clear()
        val stored = intents.findById(receipt.photoId).orElseThrow()
        assertEquals(retryAt, stored.cleanupAfter)
        assertEquals(NOW, stored.cleanupCheckedAt)
        assertNull(stored.cleanedAt)
    }

    @Test
    fun `permanent deletion reschedule fences a stale cleanup settlement`() {
        val owner = photographer()
        val event = event()
        val result = photo(owner.id, event.id)
        val receipt = intents.saveAndFlush(
            newIntent(owner.id, event.id).apply {
                state = PhotoUploadIntentState.COMMITTED
                resultPhotoId = result.id
                cleanupAfter = NOW.minusMinutes(1)
                cleanedAt = NOW.minusDays(7)
            },
        )
        val claimedUntil = NOW.plusMinutes(1)
        val deletionAt = claimedUntil

        assertEquals(1, inTransaction { intents.claimCleanup(receipt.photoId, NOW, claimedUntil) })
        assertEquals(1, inTransaction { intents.scheduleCleanupForResultPhoto(result.id, deletionAt) })
        assertEquals(0, inTransaction {
            intents.completeCleanup(receipt.photoId, claimedUntil, NOW, null, NOW.plusSeconds(20))
        })

        entityManager.clear()
        val stored = intents.findById(receipt.photoId).orElseThrow()
        assertEquals(PhotoUploadIntentState.COMMITTED, stored.state)
        assertEquals(deletionAt, stored.cleanupAfter)
        assertNull(stored.cleanedAt)
    }

    @Test
    fun `pending metrics aggregate only active unexpired receipts`() {
        val owner = photographer()
        val event = event()
        val metricNow = NOW.plusYears(1)
        intents.saveAndFlush(newIntent(
            owner.id,
            event.id,
            expiresAt = metricNow.plusHours(24),
            createdAt = metricNow.minusMinutes(10),
        ))
        intents.saveAndFlush(newIntent(
            owner.id,
            event.id,
            expiresAt = metricNow.plusHours(24),
            createdAt = metricNow,
        ).apply { state = PhotoUploadIntentState.VERIFYING })
        intents.saveAndFlush(newIntent(owner.id, event.id, expiresAt = metricNow.minusSeconds(1)))
        intents.saveAndFlush(newIntent(
            owner.id,
            event.id,
            expiresAt = metricNow.plusHours(24),
            createdAt = metricNow,
        ).apply { state = PhotoUploadIntentState.REJECTED })

        val metrics = checkNotNull(intents.pendingMetrics(metricNow))

        assertEquals(2, metrics.slots)
        assertEquals(FILE_SIZE * 2, metrics.bytes)
        assertEquals(metricNow.minusMinutes(10).toInstant(), metrics.oldestCreatedAt)
    }

    private fun photographer() = users.saveAndFlush(
        User(
            email = "intent-${UUID.randomUUID()}@test.local",
            passwordHash = "stub",
            name = "Intent Photographer",
            role = Role.PHOTOGRAPHER,
        ),
    )

    private fun event() = events.saveAndFlush(
        Event(
            slug = "intent-${UUID.randomUUID()}",
            name = "Intent Test",
            date = LocalDate.of(2026, 9, 23),
            location = "Cebu City",
            status = EventStatus.ACTIVE,
        ),
    )

    private fun photo(ownerId: UUID, eventId: UUID) = photos.saveAndFlush(
        Photo(
            eventId = eventId,
            photographerId = ownerId,
            s3Key = "events/$eventId/photos/${UUID.randomUUID()}/original.jpg",
            pricePhp = BigDecimal("100.00"),
            status = PhotoStatus.PROCESSING,
        ),
    )

    private fun intent(ownerId: UUID, eventId: UUID, requestId: UUID? = null) =
        intents.saveAndFlush(newIntent(ownerId, eventId, requestId))

    private fun newIntent(
        ownerId: UUID,
        eventId: UUID,
        requestId: UUID? = null,
        expiresAt: OffsetDateTime = NOW.plusHours(24),
        createdAt: OffsetDateTime = NOW,
    ): PhotoUploadIntent {
        val photoId = UUID.randomUUID()
        return PhotoUploadIntent(
            photoId = photoId,
            photographerId = ownerId,
            eventId = eventId,
            requestId = requestId,
            contentHash = "a".repeat(64),
            contentType = "image/jpeg",
            sizeBytes = FILE_SIZE,
            stagingKey = "staging/photo-uploads/$ownerId/$eventId/$photoId/original",
            originalKey = "events/$eventId/photos/$photoId/original.jpg",
            createdAt = createdAt,
            putExpiresAt = createdAt.plusMinutes(15),
            expiresAt = expiresAt,
        )
    }

    private fun inTransaction(block: () -> Int): Int = transactionTemplate.execute { block() } ?: 0

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2026-09-23T12:00:00Z")
        const val FILE_SIZE: Long = 8L * 1024 * 1024
    }
}
