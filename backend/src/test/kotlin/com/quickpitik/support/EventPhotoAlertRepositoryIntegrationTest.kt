package com.quickpitik.support

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.UserSelfie
import com.quickpitik.repository.EventPhotoAlertRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.email.EmailService
import com.quickpitik.service.events.EventPhotosReadyNotifier
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class EventPhotoAlertRepositoryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var alertRepository: EventPhotoAlertRepository

    @Autowired
    private lateinit var eventRepository: EventRepository

    @Autowired
    private lateinit var userRepository: UserRepository

    @Autowired
    private lateinit var userSelfieRepository: UserSelfieRepository

    @Autowired
    private lateinit var transactionTemplate: TransactionTemplate

    @Test
    fun `only one concurrent notification claim wins without marking delivery`() {
        val pending = alert(event(EventStatus.ACTIVE, LocalDate.now()).id)
        val firstClaimHeld = CountDownLatch(1)
        val releaseFirst = CountDownLatch(1)
        val secondTransactionStarted = CountDownLatch(1)
        val executor = Executors.newFixedThreadPool(2)

        try {
            val first = executor.submit<Int> {
                transactionTemplate.execute {
                    val updated = alertRepository.claimNotification(pending.id, FRESH_CLAIM, CLAIM_CUTOFF)
                    firstClaimHeld.countDown()
                    check(releaseFirst.await(5, TimeUnit.SECONDS))
                    updated
                } ?: 0
            }
            assertTrue(firstClaimHeld.await(5, TimeUnit.SECONDS))

            val second = executor.submit<Int> {
                transactionTemplate.execute {
                    secondTransactionStarted.countDown()
                    alertRepository.claimNotification(pending.id, REPLACEMENT_CLAIM, CLAIM_CUTOFF)
                } ?: 0
            }
            assertTrue(secondTransactionStarted.await(5, TimeUnit.SECONDS))

            assertFailsWith<TimeoutException> { second.get(200, TimeUnit.MILLISECONDS) }
            releaseFirst.countDown()

            assertEquals(1, first.get(5, TimeUnit.SECONDS))
            assertEquals(0, second.get(5, TimeUnit.SECONDS))
        } finally {
            releaseFirst.countDown()
            executor.shutdownNow()
        }

        val stored = alertRepository.findById(pending.id).orElseThrow()
        assertNull(stored.notifiedAt)
        assertEquals(FRESH_CLAIM, stored.notificationClaimedAt)
    }

    @Test
    fun `managed last-checked update cannot erase a concurrent claim`() {
        val pending = alert(event(EventStatus.ACTIVE, LocalDate.now()).id)

        whileManagedUpdateIsOpen(pending.id) {
            assertEquals(1, claim(pending.id, FRESH_CLAIM, CLAIM_CUTOFF))
        }

        val stored = alertRepository.findById(pending.id).orElseThrow()
        assertEquals(ENTITY_UPDATE_AT, stored.lastCheckedAt)
        assertEquals(FRESH_CLAIM, stored.notificationClaimedAt)
        assertNull(stored.notifiedAt)
    }

    @Test
    fun `managed last-checked update cannot erase a concurrent delivery stamp`() {
        val pending = alert(event(EventStatus.ACTIVE, LocalDate.now()).id)
        assertEquals(1, claim(pending.id, FRESH_CLAIM, CLAIM_CUTOFF))

        whileManagedUpdateIsOpen(pending.id) {
            assertEquals(1, stamp(pending.id, FRESH_CLAIM, DELIVERED_AT))
        }

        val stored = alertRepository.findById(pending.id).orElseThrow()
        assertEquals(ENTITY_UPDATE_AT, stored.lastCheckedAt)
        assertNull(stored.notificationClaimedAt)
        assertEquals(DELIVERED_AT, stored.notifiedAt)
    }

    @Test
    fun `failed send is reclaimed after ttl and successful retry records delivery`() {
        val clock = MutableClock(LIFECYCLE_NOW.toInstant())
        val event = event(EventStatus.ACTIVE, LocalDate.now(clock.withZone(PH_ZONE)))
        val user = userRepository.save(
            User(
                email = "lifecycle-${UUID.randomUUID()}@test.local",
                passwordHash = "stub",
                name = "Lifecycle Runner",
                role = Role.RUNNER,
            ),
        )
        val selfie = userSelfieRepository.save(
            UserSelfie(userId = user.id, s3Key = "selfies/${user.id}/lifecycle.jpg", isPrimary = true),
        )
        val pending = alertRepository.save(
            EventPhotoAlert(eventId = event.id, userId = user.id, selfieId = selfie.id),
        )
        val storage = Mockito.mock(StorageService::class.java)
        val search = Mockito.mock(PhotoSearchService::class.java)
        val email = Mockito.mock(EmailService::class.java)
        var retryClaimToken: OffsetDateTime? = null
        Mockito.`when`(storage.getBytes(selfie.s3Key)).thenReturn(byteArrayOf(1, 2, 3))
        Mockito.`when`(
            search.searchByFace(
                anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyBoolean(),
            ),
        ).thenReturn(PaginatedResponse(emptyList<PhotoDto>(), 1L, 0, 1))
        Mockito.doThrow(RuntimeException("resend down")).doAnswer {
            retryClaimToken = alertRepository.findById(pending.id).orElseThrow().notificationClaimedAt
            null
        }.`when`(email)
            .sendEventPhotosReadyEmail(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())
        val notifier = EventPhotosReadyNotifier(
            alertRepository,
            eventRepository,
            userRepository,
            userSelfieRepository,
            storage,
            search,
            AiApiProperties(enabled = true, photosReadyClaimTtl = CLAIM_TTL),
            email,
            transactionTemplate,
            clock,
        )

        notifier.notifyIfMatched(pending.id)

        var stored = alertRepository.findById(pending.id).orElseThrow()
        assertEquals(LIFECYCLE_NOW, stored.notificationClaimedAt)
        assertNull(stored.notifiedAt)
        assertFalse(pendingAt(clock, pending.id))

        clock.advance(CLAIM_TTL.plusSeconds(1))
        assertTrue(pendingAt(clock, pending.id))

        notifier.notifyIfMatched(pending.id)

        stored = alertRepository.findById(pending.id).orElseThrow()
        assertEquals(OffsetDateTime.ofInstant(clock.instant(), ZoneOffset.UTC), retryClaimToken)
        assertNull(stored.notificationClaimedAt)
        assertEquals(OffsetDateTime.ofInstant(clock.instant(), ZoneOffset.UTC), stored.notifiedAt)
        assertFalse(pendingAt(clock, pending.id))
    }

    @Test
    fun `pending sweep covers completed grace-day events and rotates checked alerts`() {
        val today = LocalDate.now()
        val active = event(EventStatus.ACTIVE, today)
        val completedGraceDay = event(EventStatus.COMPLETED, today.minusDays(4))
        val archived = event(EventStatus.ARCHIVED, today)
        val expired = event(EventStatus.ACTIVE, today.minusDays(5))

        val neverChecked = alert(active.id)
        val alreadyChecked = alert(active.id, OffsetDateTime.now().minusHours(1))
        val completedAlert = alert(completedGraceDay.id)
        val archivedAlert = alert(archived.id)
        val expiredAlert = alert(expired.id)

        val pending = alertRepository.findPendingInWindow(
            today,
            today.minusDays(4),
            CLAIM_CUTOFF,
            PageRequest.of(0, 100),
        ).map { it.id }

        assertContains(pending, neverChecked.id)
        assertContains(pending, completedAlert.id)
        assertFalse(archivedAlert.id in pending)
        assertFalse(expiredAlert.id in pending)
        assertTrue(pending.indexOf(neverChecked.id) < pending.indexOf(alreadyChecked.id))
    }

    @Test
    fun `fresh leases are skipped while expired leases are selected and reclaimed`() {
        val event = event(EventStatus.ACTIVE, LocalDate.now())
        val fresh = alert(event.id, notificationClaimedAt = FRESH_CLAIM)
        val expired = alert(event.id, notificationClaimedAt = OLD_CLAIM)
        val unclaimed = alert(event.id)

        val pending = pending(LocalDate.now()).map { it.id }

        assertFalse(fresh.id in pending)
        assertContains(pending, expired.id)
        assertContains(pending, unclaimed.id)
        assertEquals(1, claim(expired.id, REPLACEMENT_CLAIM, CLAIM_CUTOFF))
        assertEquals(REPLACEMENT_CLAIM, alertRepository.findById(expired.id).orElseThrow().notificationClaimedAt)
    }

    @Test
    fun `an expired crash claim remains recoverable after the event window`() {
        val today = LocalDate.now()
        val oldEvent = event(EventStatus.COMPLETED, today.minusDays(5))
        val crashed = alert(oldEvent.id, notificationClaimedAt = OLD_CLAIM)
        val neverClaimed = alert(oldEvent.id)

        val pending = pending(today).map { it.id }

        assertContains(pending, crashed.id)
        assertFalse(neverClaimed.id in pending)
    }

    @Test
    fun `a stale worker cannot stamp or clear a replacement claim`() {
        val pending = alert(event(EventStatus.ACTIVE, LocalDate.now()).id, notificationClaimedAt = OLD_CLAIM)
        assertEquals(1, claim(pending.id, REPLACEMENT_CLAIM, CLAIM_CUTOFF))

        assertEquals(0, stamp(pending.id, OLD_CLAIM, DELIVERED_AT))
        var stored = alertRepository.findById(pending.id).orElseThrow()
        assertNull(stored.notifiedAt)
        assertEquals(REPLACEMENT_CLAIM, stored.notificationClaimedAt)

        assertEquals(1, stamp(pending.id, REPLACEMENT_CLAIM, DELIVERED_AT))
        stored = alertRepository.findById(pending.id).orElseThrow()
        assertEquals(DELIVERED_AT, stored.notifiedAt)
        assertNull(stored.notificationClaimedAt)
    }

    @Test
    fun `pending selection excludes delivered deleted hidden and opted-out alerts`() {
        val today = LocalDate.now()
        val visibleEvent = event(EventStatus.ACTIVE, today)
        val eligible = alert(visibleEvent.id)
        val delivered = alert(visibleEvent.id, notifiedAt = DELIVERED_AT)

        val deletedEvent = event(EventStatus.ACTIVE, today).also {
            it.deletedAt = OffsetDateTime.now()
            eventRepository.saveAndFlush(it)
        }
        val deleted = alert(deletedEvent.id)
        val hidden = alert(event(EventStatus.ARCHIVED, today).id)
        val optedOut = alert(visibleEvent.id)
        alertRepository.delete(optedOut)
        alertRepository.flush()

        val pending = pending(today).map { it.id }

        assertContains(pending, eligible.id)
        assertFalse(delivered.id in pending)
        assertFalse(deleted.id in pending)
        assertFalse(hidden.id in pending)
        assertFalse(optedOut.id in pending)
    }

    private fun event(status: EventStatus, date: LocalDate): Event = eventRepository.save(
        Event(
            slug = "photo-alert-${UUID.randomUUID()}",
            name = "Photo Alert Test",
            date = date,
            location = "Cebu City",
            status = status,
        ),
    )

    private fun alert(
        eventId: UUID,
        lastCheckedAt: OffsetDateTime? = null,
        notificationClaimedAt: OffsetDateTime? = null,
        notifiedAt: OffsetDateTime? = null,
    ): EventPhotoAlert =
        alertRepository.save(
            EventPhotoAlert(
                eventId = eventId,
                userId = userRepository.save(
                    User(
                        email = "runner-${UUID.randomUUID()}@test.local",
                        passwordHash = "stub",
                        name = "Test Runner",
                        role = Role.RUNNER,
                    ),
                ).id,
                lastCheckedAt = lastCheckedAt,
                notificationClaimedAt = notificationClaimedAt,
                notifiedAt = notifiedAt,
            ),
        )

    private fun pending(today: LocalDate): List<EventPhotoAlert> =
        alertRepository.findPendingInWindow(today, today.minusDays(4), CLAIM_CUTOFF, PageRequest.of(0, 100))

    private fun claim(id: UUID, token: OffsetDateTime, expiredBefore: OffsetDateTime): Int =
        transactionTemplate.execute {
            alertRepository.claimNotification(id, token, expiredBefore)
        } ?: 0

    private fun stamp(id: UUID, token: OffsetDateTime, deliveredAt: OffsetDateTime): Int =
        transactionTemplate.execute {
            alertRepository.stampNotificationDelivered(id, token, deliveredAt)
        } ?: 0

    private fun pendingAt(clock: Clock, id: UUID): Boolean {
        val now = OffsetDateTime.ofInstant(clock.instant(), ZoneOffset.UTC)
        val today = LocalDate.now(clock.withZone(PH_ZONE))
        return id in alertRepository.findPendingInWindow(
            today,
            today.minusDays(4),
            now.minus(CLAIM_TTL),
            PageRequest.of(0, 100),
        ).map { it.id }
    }

    private fun whileManagedUpdateIsOpen(id: UUID, interleavingUpdate: () -> Unit) {
        val entityLoaded = CountDownLatch(1)
        val allowFlush = CountDownLatch(1)
        val executor = Executors.newSingleThreadExecutor()
        try {
            val entityUpdate = executor.submit {
                transactionTemplate.executeWithoutResult {
                    val managed = alertRepository.findById(id).orElseThrow()
                    managed.lastCheckedAt = ENTITY_UPDATE_AT
                    entityLoaded.countDown()
                    check(allowFlush.await(5, TimeUnit.SECONDS))
                }
            }
            assertTrue(entityLoaded.await(5, TimeUnit.SECONDS))
            interleavingUpdate()
            allowFlush.countDown()
            entityUpdate.get(5, TimeUnit.SECONDS)
        } finally {
            allowFlush.countDown()
            executor.shutdownNow()
        }
    }

    private class MutableClock(private var current: Instant) : Clock() {
        override fun getZone(): ZoneId = ZoneOffset.UTC
        override fun withZone(zone: ZoneId): Clock = Clock.fixed(current, zone)
        override fun instant(): Instant = current
        fun advance(duration: Duration) {
            current = current.plus(duration)
        }
    }

    private fun <T> anyArg(): T = Mockito.any()

    private companion object {
        val OLD_CLAIM: OffsetDateTime = OffsetDateTime.parse("2026-09-22T03:00:00.123456Z")
        val CLAIM_CUTOFF: OffsetDateTime = OffsetDateTime.parse("2026-09-22T04:00:00.123456Z")
        val FRESH_CLAIM: OffsetDateTime = OffsetDateTime.parse("2026-09-22T04:05:00.123456Z")
        val REPLACEMENT_CLAIM: OffsetDateTime = OffsetDateTime.parse("2026-09-22T04:10:00.123456Z")
        val DELIVERED_AT: OffsetDateTime = OffsetDateTime.parse("2026-09-22T04:11:00.123456Z")
        val ENTITY_UPDATE_AT: OffsetDateTime = OffsetDateTime.parse("2026-09-22T04:12:00.123456Z")
        val LIFECYCLE_NOW: OffsetDateTime = OffsetDateTime.parse("2026-09-22T16:10:00.123456Z")
        val CLAIM_TTL: Duration = Duration.ofMinutes(10)
        val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")
    }
}
