package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.flagReasonCodesMatching
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// AX-10: the payouts list takes several statuses at once, both lists have per-status counts
// that equal the list's total, and the flag search reaches photographer and reporter.
class AdminPayoutFlagQueryIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var cycles: PayoutCycleRepository
    @Autowired private lateinit var flags: FlagRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var settings: PhotographerSettingsRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var photos: PhotoRepository

    @Test
    fun `payout status list returns exactly those statuses and counts agree`() {
        val marker = "pay-ax10-${UUID.randomUUID().toString().take(8)}"
        val byStatus = listOf(
            PayoutCycleStatus.PENDING, PayoutCycleStatus.APPROVED, PayoutCycleStatus.APPROVED,
            PayoutCycleStatus.PROCESSING, PayoutCycleStatus.PAID, PayoutCycleStatus.FAILED,
        ).mapIndexed { i, s ->
            // One photographer per cycle: a photographer may have only one open payout (uq_payout_open_per_photographer).
            s.wire to cycles.saveAndFlush(PayoutCycle(id = "$marker-$i".uppercase(), photographerId = user(Role.PHOTOGRAPHER, "Payout fixture").id,
                weekOf = LocalDate.of(2026, 9, 1).plusWeeks(i.toLong()), statusWire = s.wire)).id
        }

        fun ids(statuses: List<String>) = cycles.pageForAdmin(statuses, marker, PageRequest.of(0, 50))

        val inProgress = ids(listOf("approved", "processing"))
        assertEquals(byStatus.filter { it.first in setOf("approved", "processing") }.map { it.second }.toSet(),
            inProgress.content.map { it.id }.toSet())
        assertEquals(3, inProgress.totalElements)
        assertEquals(6, ids(PayoutCycleStatus.entries.map { it.wire }).totalElements)

        val counts = cycles.countByStatusForAdmin(marker).associate { it.status to it.total }
        assertEquals(mapOf("pending" to 1L, "approved" to 2L, "processing" to 1L, "paid" to 1L, "failed" to 1L), counts)
    }

    // FT-28: the payout search also finds the photographer by name, brand name or handle
    // (typed with or without "@"); % and _ stay plain text; the tab counts follow the list.
    @Test
    fun `payout search finds the photographer by name, brand or handle, and the counts follow`() {
        val t = UUID.randomUUID().toString().take(8)
        val owner = user(Role.PHOTOGRAPHER, "Ft28 Juan $t")
        settings.saveAndFlush(PhotographerSettings(userId = owner.id, handle = "ft28shots-$t", brandName = "Ft28 Studio $t"))
        val noSettings = user(Role.PHOTOGRAPHER, "Ft28 50%_a $t")
        val lookAlike = user(Role.PHOTOGRAPHER, "Ft28 50xxa $t")
        val pending = payout("FT28-$t-1", owner.id, PayoutCycleStatus.PENDING)
        val paid = payout("FT28-$t-2", owner.id, PayoutCycleStatus.PAID)
        val wildcard = payout("FT28-$t-3", noSettings.id, PayoutCycleStatus.FAILED)
        payout("FT28-$t-4", lookAlike.id, PayoutCycleStatus.FAILED)

        // The service hands the repository a trimmed, lower-cased, LIKE-escaped term.
        fun find(term: String) =
            cycles.pageForAdmin(PayoutCycleStatus.entries.map { it.wire }, term, PageRequest.of(0, 50)).content.map { it.id }.toSet()

        assertEquals(setOf(pending, paid), find("ft28shots-$t"))
        assertEquals(setOf(pending, paid), find("@ft28shots-$t"))
        assertEquals(setOf(pending, paid), find("ft28 studio $t"))
        assertEquals(setOf(pending, paid), find("ft28 juan $t"))
        assertEquals(setOf(wildcard), find("50\\%\\_a $t"))
        assertEquals(setOf(paid), find("ft28-$t-2"))

        val counts = cycles.countByStatusForAdmin("@ft28shots-$t").associate { it.status to it.total }
        assertEquals(mapOf("pending" to 1L, "paid" to 1L), counts)
    }

    @Test
    fun `flag search finds photographer handle, reporter name and handle, and the exact flag id`() {
        val t = UUID.randomUUID().toString().take(8)
        val photographer = user(Role.PHOTOGRAPHER, "Owner $t")
        settings.saveAndFlush(PhotographerSettings(userId = photographer.id, handle = "ax10owner-$t"))
        val reporter = user(Role.PHOTOGRAPHER, "Reporter ax10name-$t")
        settings.saveAndFlush(PhotographerSettings(userId = reporter.id, handle = "ax10rep-$t"))
        val event = events.saveAndFlush(Event(slug = "ax10-${UUID.randomUUID()}", name = "Run", date = LocalDate.of(2026, 10, 1),
            location = "Cebu", status = EventStatus.ACTIVE, createdBy = photographer.id))
        val photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = photographer.id,
            s3Key = "ax10/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("100.00")))

        val onPhoto = flag("photo", photo.id, reporter = reporter.id)
        val onEvent = flag("event", event.id, status = FlagStatus.RESOLVED)
        val onUser = flag("user", photographer.id, status = FlagStatus.DISMISSED)
        val unrelated = flag("user", user(Role.PHOTOGRAPHER, "Someone else").id)

        assertEquals(setOf(onPhoto, onEvent, onUser), search("AX10OWNER-$t"))
        assertEquals(setOf(onPhoto), search("ax10name-$t"))
        assertEquals(setOf(onPhoto), search("ax10rep-$t"))
        assertEquals(setOf(unrelated), search(unrelated.toString().uppercase()))
        assertEquals(setOf(onEvent), flags.pageForAdmin("resolved", "ax10owner-$t", emptyList(), PageRequest.of(0, 50)).content.map { it.id }.toSet())

        val counts = flags.countByStatusForAdmin("ax10owner-$t", emptyList()).associate { it.status to it.total }
        assertEquals(mapOf("open" to 1L, "resolved" to 1L, "dismissed" to 1L), counts)
    }

    // AX-48: History search finds a flag by what its row shows: a handle-less photographer's
    // name, the reporter's email, and the reason label; a reason code still works.
    @Test
    fun `flag search finds a no-handle photographer by name, a reporter by email, and a reason by its label`() {
        val t = UUID.randomUUID().toString().take(8)
        val photographer = user(Role.PHOTOGRAPHER, "Ax48 Nohandle $t")
        val event = events.saveAndFlush(Event(slug = "ax48-${UUID.randomUUID()}", name = "Run", date = LocalDate.of(2026, 10, 1),
            location = "Cebu", status = EventStatus.ACTIVE, eventType = RaceType.FUN_RUN, submittedBy = photographer.id))
        val photo = photos.saveAndFlush(Photo(eventId = event.id, photographerId = photographer.id,
            s3Key = "ax48/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("100.00")))
        val reporter = user(Role.RUNNER, "Runner $t")

        val onPhoto = flag("photo", photo.id, reporter = reporter.id, status = FlagStatus.RESOLVED, reason = "inappropriate")
        val onEvent = flag("event", event.id, status = FlagStatus.DISMISSED, reason = "low_quality")
        val onUser = flag("user", photographer.id, status = FlagStatus.HIDDEN)

        assertEquals(setOf(onPhoto, onEvent, onUser), search("nohandle $t"))
        assertEquals(setOf(onPhoto), search(reporter.email.uppercase()))
        assertTrue(onPhoto in search("NSFW"))
        assertTrue(onEvent in search("low quality"))
        assertTrue(onEvent in search("low_quality"))
    }

    // The reason codes come from the label table, as AdminFlagService passes them.
    private fun search(q: String): Set<UUID> =
        flags.pageForAdmin(null, q, flagReasonCodesMatching(q), PageRequest.of(0, 50)).content.map { it.id }.toSet()

    private fun flag(kind: String, target: UUID, reporter: UUID? = null, status: FlagStatus = FlagStatus.OPEN, reason: String = "other"): UUID =
        flags.saveAndFlush(Flag(targetKindWire = kind, targetId = target, reporterId = reporter, reason = reason,
            statusWire = status.wire)).id

    private fun payout(id: String, photographer: UUID, status: PayoutCycleStatus): String =
        cycles.saveAndFlush(PayoutCycle(id = id, photographerId = photographer, weekOf = LocalDate.of(2026, 9, 1), statusWire = status.wire)).id

    private fun user(role: Role, name: String) = users.saveAndFlush(
        User(email = "ax10-${UUID.randomUUID()}@test.local", passwordHash = "test-only-unused", name = name, role = role),
    )
}
