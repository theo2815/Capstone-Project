package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.RaceProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.Role
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.SystemFlagService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// AA-R9: a signed-in user reports an event or a photographer. Three different
// accounts hide an unproven typed race; everything else is only flagged.
class RaceReportServiceTest {

    private lateinit var eventRepository: EventRepository
    private lateinit var settingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var flagRepository: FlagRepository
    private lateinit var flags: SystemFlagService
    private lateinit var raceProof: RaceProofService

    // (kind, targetId, reason, reporterId) of every raise the service made.
    private val raised = mutableListOf<List<Any?>>()
    private val openReports = mutableSetOf<Pair<UUID, UUID?>>()
    private var distinctReporters = 0L
    private var proven = false

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        flagRepository = Mockito.mock(FlagRepository::class.java)
        flags = Mockito.mock(SystemFlagService::class.java)
        raceProof = Mockito.mock(RaceProofService::class.java)
        // Dedupe like SystemFlagService: one open row per (target, reporter) for user reports.
        Mockito.`when`(flags.raise(anyArg(), anyArg(), anyArg(), Mockito.any(), Mockito.any())).thenAnswer { inv ->
            val target = inv.getArgument<UUID>(1)
            val reason = inv.getArgument<SystemFlagReason>(2)
            val reporter = inv.getArgument<UUID?>(4)
            raised += listOf(inv.getArgument<FlagTargetKind>(0), target, reason, reporter)
            reason != SystemFlagReason.USER_REPORT || openReports.add(target to reporter)
        }
        Mockito.`when`(flagRepository.countDistinctReporters(anyArg(), anyArg())).thenAnswer { distinctReporters }
        Mockito.`when`(raceProof.isProven(anyArg())).thenAnswer { proven }
        Mockito.`when`(eventRepository.hideUnprovenRace(anyArg(), anyArg())).thenAnswer { inv ->
            val event = events.getValue(inv.getArgument(0))
            if (event.hiddenAt != null) 0 else 1.also { event.hiddenAt = inv.getArgument(1) }
        }
        Mockito.`when`(eventRepository.findById(anyArg())).thenAnswer { inv ->
            Optional.ofNullable(events[inv.getArgument(0)])
        }
    }

    private val events = mutableMapOf<UUID, Event>()

    private fun service() = RaceReportService(
        eventRepository, settingsRepository, userRepository, flagRepository, flags, raceProof, RaceProperties(), CLOCK,
    )

    private fun event(type: RaceType? = RaceType.ROAD_RACE, status: EventStatus = EventStatus.ACTIVE) = Event(
        slug = "rr-${UUID.randomUUID().toString().take(6)}",
        name = "Cebu Bay Run",
        date = LocalDate.of(2026, 11, 1),
        location = "Cebu City",
        status = status,
        eventType = type,
    ).also { events[it.id] = it }

    // Each account reports once; the count answers what the database would see afterwards.
    private fun reportBy(n: Int, event: Event) = List(n) { i ->
        distinctReporters = (i + 1).toLong()
        service().reportEvent(UUID.randomUUID(), event.id, NOTE)
    }

    private fun reasons() = raised.map { it[2] }

    @Test
    fun `a second report by the same account is a no-op`() {
        val e = event()
        val me = UUID.randomUUID()
        distinctReporters = 1
        val first = service().reportEvent(me, e.id, NOTE)
        val second = service().reportEvent(me, e.id, NOTE)
        assertFalse(first.alreadyReported)
        assertTrue(second.alreadyReported)
        assertTrue(second.reported)
        assertNull(e.hiddenAt)
    }

    @Test
    fun `three accounts hide an unproven race with a flag`() {
        val e = event()
        val results = reportBy(3, e)
        assertTrue(results.none { it.alreadyReported })
        assertNotNull(e.hiddenAt)
        assertEquals(
            listOf(SystemFlagReason.USER_REPORT, SystemFlagReason.USER_REPORT, SystemFlagReason.USER_REPORT,
                SystemFlagReason.REPORT_THRESHOLD_HIDDEN),
            reasons(),
        )
        // The threshold flag is the system's own finding: no reporter.
        assertEquals(listOf(FlagTargetKind.EVENT, e.id, SystemFlagReason.REPORT_THRESHOLD_HIDDEN, null), raised.last())
    }

    @Test
    fun `two accounts do not hide a race`() {
        val e = event()
        reportBy(2, e)
        assertNull(e.hiddenAt)
        assertFalse(SystemFlagReason.REPORT_THRESHOLD_HIDDEN in reasons())
    }

    @Test
    fun `the same account reporting twice at once counts once`() {
        // Two rows slipped past the read-then-insert dedupe; the database still counts one account.
        val e = event()
        Mockito.`when`(flags.raise(anyArg(), anyArg(), anyArg(), Mockito.any(), Mockito.any())).thenReturn(true)
        distinctReporters = 1
        repeat(3) { service().reportEvent(ME, e.id, NOTE) }
        assertNull(e.hiddenAt)
    }

    @Test
    fun `a proven race is only flagged`() {
        val e = event()
        proven = true
        reportBy(3, e)
        assertNull(e.hiddenAt)
        assertEquals(List(3) { SystemFlagReason.USER_REPORT }, reasons())
    }

    @Test
    fun `an untyped admin event is never hidden`() {
        val e = event(type = null)
        reportBy(5, e)
        assertNull(e.hiddenAt)
        assertEquals(List(5) { SystemFlagReason.USER_REPORT }, reasons())
        Mockito.verify(eventRepository, Mockito.never()).hideUnprovenRace(anyArg(), anyArg())
    }

    @Test
    fun `an already hidden race gets no second threshold flag`() {
        val e = event().also { it.hiddenAt = CLOCK.instant().atOffset(ZoneOffset.UTC) }
        reportBy(4, e)
        assertFalse(SystemFlagReason.REPORT_THRESHOLD_HIDDEN in reasons())
    }

    @Test
    fun `a draft event answers not found`() {
        val e = event(status = EventStatus.DRAFT)
        val ex = assertFailsWith<NotFoundException> { service().reportEvent(ME, e.id, NOTE) }
        assertEquals(ErrorCodes.EVENT_NOT_FOUND, ex.code)
        assertTrue(raised.isEmpty())
    }

    @Test
    fun `a deleted or unknown event answers not found`() {
        val gone = event().also { it.deletedAt = CLOCK.instant().atOffset(ZoneOffset.UTC) }
        listOf(gone.id, UUID.randomUUID()).forEach { id ->
            assertEquals(ErrorCodes.EVENT_NOT_FOUND, assertFailsWith<NotFoundException> { service().reportEvent(ME, id, NOTE) }.code)
        }
        assertTrue(raised.isEmpty())
    }

    @Test
    fun `a photographer is reported by handle`() {
        val photographer = User(email = "p@test.local", passwordHash = "x", name = "P", role = Role.PHOTOGRAPHER)
        Mockito.`when`(settingsRepository.findByHandleIgnoreCase("lens.cebu")).thenReturn(
            PhotographerSettings(userId = photographer.id, verificationStatus = VerificationStatus.APPROVED),
        )
        Mockito.`when`(userRepository.findById(photographer.id)).thenReturn(Optional.of(photographer))

        val first = service().reportPhotographer(ME, " Lens.Cebu ", NOTE)
        val again = service().reportPhotographer(ME, "lens.cebu", NOTE)

        assertFalse(first.alreadyReported)
        assertTrue(again.alreadyReported)
        assertEquals(listOf(FlagTargetKind.USER, photographer.id, SystemFlagReason.USER_REPORT, ME), raised.first())
    }

    @Test
    fun `an unknown handle or a non-photographer answers user not found`() {
        val runner = User(email = "r@test.local", passwordHash = "x", name = "R", role = Role.RUNNER)
        Mockito.`when`(settingsRepository.findByHandleIgnoreCase("runner")).thenReturn(
            PhotographerSettings(userId = runner.id, verificationStatus = VerificationStatus.INCOMPLETE),
        )
        Mockito.`when`(userRepository.findById(runner.id)).thenReturn(Optional.of(runner))
        listOf("nobody", "runner").forEach { handle ->
            val ex = assertFailsWith<NotFoundException> { service().reportPhotographer(ME, handle, NOTE) }
            assertEquals(ErrorCodes.USER_NOT_FOUND, ex.code)
        }
        assertTrue(raised.isEmpty())
    }

    @Test
    fun `a photographer cannot report a race they submitted`() {
        val e = event().also { it.submittedBy = ME }
        val ex = assertFailsWith<ApiException> { service().reportEvent(ME, e.id, NOTE) }
        assertEquals(HttpStatus.FORBIDDEN, ex.status)
        assertEquals(ErrorCodes.FORBIDDEN, ex.code)
        Mockito.verify(flags, Mockito.never()).raise(anyArg(), anyArg(), anyArg(), Mockito.any(), Mockito.any())
    }

    @Test
    fun `a creator cannot report their own event`() {
        val e = event().also { it.createdBy = ME }
        val ex = assertFailsWith<ApiException> { service().reportEvent(ME, e.id, NOTE) }
        assertEquals(HttpStatus.FORBIDDEN, ex.status)
        assertEquals(ErrorCodes.FORBIDDEN, ex.code)
        Mockito.verify(flags, Mockito.never()).raise(anyArg(), anyArg(), anyArg(), Mockito.any(), Mockito.any())
    }

    @Test
    fun `a photographer cannot report their own profile`() {
        val me = User(id = ME, email = "me@test.local", passwordHash = "x", name = "Me", role = Role.PHOTOGRAPHER)
        Mockito.`when`(settingsRepository.findByHandleIgnoreCase("my.handle")).thenReturn(
            PhotographerSettings(userId = me.id, verificationStatus = VerificationStatus.APPROVED),
        )
        Mockito.`when`(userRepository.findById(me.id)).thenReturn(Optional.of(me))
        val ex = assertFailsWith<ApiException> { service().reportPhotographer(ME, "my.handle", NOTE) }
        assertEquals(HttpStatus.FORBIDDEN, ex.status)
        assertEquals(ErrorCodes.FORBIDDEN, ex.code)
        Mockito.verify(flags, Mockito.never()).raise(anyArg(), anyArg(), anyArg(), Mockito.any(), Mockito.any())
    }

    private fun <T> anyArg(): T = Mockito.any()

    private companion object {
        val CLOCK: Clock = Clock.fixed(Instant.parse("2026-11-10T04:00:00Z"), ZoneOffset.UTC)
        val ME: UUID = UUID.randomUUID()
        const val NOTE = "This race never happened, the photos are from 2024."
    }
}
