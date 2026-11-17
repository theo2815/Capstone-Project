package com.quickpitik.service.events

import com.quickpitik.config.RaceProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RaceProofCounts
import com.quickpitik.repository.SharedBibEvent
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.Pageable
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// AA-R8 (Net 2): when uploads close, an unproven typed race is hidden with a flag.
class RaceProofServiceTest {

    private lateinit var eventRepository: EventRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var flags: SystemFlagService
    private val races = mutableListOf<Event>()
    private val counts = mutableMapOf<UUID, RaceProofCounts>()
    private val sharedBibs = mutableMapOf<UUID, List<SharedBibEvent>>()
    private val raised = mutableListOf<Triple<UUID, SystemFlagReason, String?>>()
    private var relistBeforeHide = false

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        flags = Mockito.mock(SystemFlagService::class.java)
        // Answer like the database would, from the in-memory rows.
        Mockito.`when`(eventRepository.findRacesDueForProof(anyArg(), anyArg())).thenAnswer { inv ->
            val closed = inv.getArgument<LocalDate>(0)
            races.filter { it.proofCheckedAt == null && !it.date.isAfter(closed) }
        }
        Mockito.`when`(eventRepository.claimProofCheck(anyArg(), anyArg())).thenAnswer { inv ->
            val race = races.first { it.id == inv.getArgument<UUID>(0) }
            if (race.proofCheckedAt != null) 0 else 1.also { race.proofCheckedAt = inv.getArgument(1) }
        }
        Mockito.`when`(eventRepository.hideUnprovenRace(anyArg(), anyArg())).thenAnswer { inv ->
            val race = races.first { it.id == inv.getArgument<UUID>(0) }
            if (relistBeforeHide) race.adminApprovedAt = OffsetDateTime.now(CLOCK)
            if (race.hiddenAt != null || race.adminApprovedAt != null) 0 else 1.also { race.hiddenAt = inv.getArgument(1) }
        }
        Mockito.`when`(photoRepository.raceProofCounts(anyArg())).thenAnswer { inv ->
            counts[inv.getArgument(0)] ?: counts(0, 0, 0, 0)
        }
        Mockito.`when`(photoRepository.eventsSharingBibs(anyArg(), anyArg(), anyArg(), Mockito.anyInt()))
            .thenAnswer { inv -> sharedBibs[inv.getArgument(0)].orEmpty() }
        Mockito.`when`(flags.raise(anyArg(), anyArg(), anyArg(), Mockito.any(), Mockito.isNull())).thenAnswer { inv ->
            raised += Triple(inv.getArgument(1), inv.getArgument(2), inv.getArgument(3))
            true
        }
    }

    private fun service() = RaceProofService(
        eventRepository, photoRepository, flags, RaceProperties(), CLOCK, testTransactionTemplate(),
    )

    private fun race(date: LocalDate = CLOSED, type: RaceType = RaceType.ROAD_RACE) = Event(
        slug = "race-${UUID.randomUUID().toString().take(6)}",
        name = "Cebu Bay Run",
        date = date,
        location = "Cebu City",
        status = EventStatus.ACTIVE,
        eventType = type,
    ).also { races += it }

    private fun counts(photographers: Long, total: Long, indexed: Long, withBib: Long) = object : RaceProofCounts {
        override val photographers = photographers
        override val total = total
        override val indexed = indexed
        override val withBib = withBib
    }

    private fun reasons() = raised.map { it.second }

    @Test
    fun `a race still in its upload window is untouched`() {
        val open = race(date = CLOSED.plusDays(1))
        service().sweep()
        assertNull(open.proofCheckedAt)
        assertNull(open.hiddenAt)
        assertTrue(raised.isEmpty())
    }

    @Test
    fun `two photographers prove it`() {
        val r = race()
        counts[r.id] = counts(2, 4, 0, 0)
        service().sweep()
        assertNotNull(r.proofCheckedAt)
        assertNull(r.hiddenAt)
        assertTrue(raised.isEmpty())
    }

    @Test
    fun `twenty five indexed with forty percent bibs proves it`() {
        val r = race()
        counts[r.id] = counts(1, 25, 25, 10)
        service().sweep()
        assertNull(r.hiddenAt)
        assertTrue(raised.isEmpty())
    }

    @Test
    fun `one photographer and few bibs hides it with a flag`() {
        val r = race()
        counts[r.id] = counts(1, 25, 25, 2)
        service().sweep()
        assertNotNull(r.hiddenAt)
        assertEquals(listOf(SystemFlagReason.NO_PROOF_HIDDEN), reasons())
        assertEquals(r.id, raised.single().first)
    }

    @Test
    fun `mostly skipped photos only flag it`() {
        val r = race()
        counts[r.id] = counts(1, 30, 5, 5)
        service().sweep()
        assertNull(r.hiddenAt)
        assertEquals(listOf(SystemFlagReason.NO_PROOF_UNINDEXED), reasons())
    }

    @Test
    fun `a one-photographer cycling race with few bibs is only flagged`() {
        val r = race(type = RaceType.CYCLING)
        counts[r.id] = counts(1, 25, 25, 2)
        service().sweep()
        assertNull(r.hiddenAt)
        assertEquals(listOf(SystemFlagReason.NO_PROOF_UNINDEXED), reasons())
    }

    @Test
    fun `zero photos hides it`() {
        val r = race(type = RaceType.CYCLING)
        service().sweep()
        assertNotNull(r.hiddenAt)
        assertEquals(listOf(SystemFlagReason.NO_PROOF_HIDDEN), reasons())
    }

    @Test
    fun `an admin approved race is never hidden`() {
        val r = race().also { it.adminApprovedAt = OffsetDateTime.now(CLOCK) }
        service().sweep()
        assertNull(r.hiddenAt)
        assertTrue(raised.isEmpty())
    }

    @Test
    fun `a second run does nothing`() {
        val r = race()
        service().sweep()
        service().sweep()
        assertEquals(listOf(SystemFlagReason.NO_PROOF_HIDDEN), reasons())
        assertNotNull(r.hiddenAt)
    }

    @Test
    fun `an admin relist between claim and hide wins`() {
        val r = race()
        relistBeforeHide = true
        service().sweep()
        assertNull(r.hiddenAt)
        assertTrue(raised.isEmpty())
    }

    @Test
    fun `two same-bib matches make one flag naming both`() {
        val r = race()
        counts[r.id] = counts(2, 60, 60, 60)
        val a = UUID.randomUUID()
        val b = UUID.randomUUID()
        sharedBibs[r.id] = listOf(shared(a, "Cebu Marathon", 41), shared(b, "Cebu City Marathon", 30))
        service().sweep()
        val flag = raised.single()
        assertEquals(SystemFlagReason.SAME_BIB_DUPLICATE, flag.second)
        assertEquals("Cebu Marathon ($a): 41 bibs\nCebu City Marathon ($b): 30 bibs", flag.third)
    }

    @Test
    fun `an open same-bib flag gets the new matches appended`() {
        val r = race()
        counts[r.id] = counts(2, 60, 60, 60)
        val a = UUID.randomUUID()
        sharedBibs[r.id] = listOf(shared(a, "Cebu Marathon", 41))
        Mockito.`when`(flags.raise(anyArg(), anyArg(), anyArg(), Mockito.any(), Mockito.isNull())).thenReturn(false)
        service().sweep()
        Mockito.verify(flags).appendNote(
            FlagTargetKind.EVENT, r.id, SystemFlagReason.SAME_BIB_DUPLICATE,
            "${OffsetDateTime.now(CLOCK)} Cebu Marathon ($a): 41 bibs",
        )
    }

    private fun shared(id: UUID, name: String, n: Long) = object : SharedBibEvent {
        override val eventId = id
        override val name = name
        override val shared = n
    }

    private fun <T> anyArg(): T = Mockito.any()

    private companion object {
        // 2026-11-20 10:00 in Manila: uploads for races dated 2026-11-16 or earlier are closed.
        val CLOCK: Clock = Clock.fixed(Instant.parse("2026-11-20T02:00:00Z"), ZoneOffset.UTC)
        val CLOSED: LocalDate = LocalDate.of(2026, 11, 16)
    }
}
