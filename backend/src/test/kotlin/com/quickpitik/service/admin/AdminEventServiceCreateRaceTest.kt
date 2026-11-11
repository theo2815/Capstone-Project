package com.quickpitik.service.admin

import com.quickpitik.common.IdempotencyKey
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.entity.AdminActionOperation
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Event
import com.quickpitik.entity.RaceType
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.events.EventFaceExpiry
import com.quickpitik.service.events.RaceEventService
import com.quickpitik.service.events.RaceInput
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// AA-R7: an admin-created race carries a type, is keyed and admin approved at birth, and is
// inserted under the same race-creation lock as photographer creates, on both create paths.
class AdminEventServiceCreateRaceTest {
    private val adminId = UUID.randomUUID()
    private val events = Mockito.mock(EventRepository::class.java)
    private val commands = Mockito.mock(AdminActionCommandService::class.java)
    private val covers = Mockito.mock(EventCoverService::class.java)
    private val decisions = Mockito.mock(AdminDecisionLogService::class.java)
    private val raceEvents = Mockito.mock(RaceEventService::class.java)
    private var saved: Event? = null
    private var hashedFields: List<String>? = null
    private lateinit var service: AdminEventService

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> anyLike(dummy: T): T { Mockito.any<T>(); return dummy }

    @BeforeEach
    fun setUp() {
        Mockito.`when`(events.save(anyArg<Event>())).thenAnswer { it.getArgument<Event>(0).also { e -> saved = e } }
        Mockito.`when`(events.findById(anyArg())).thenAnswer { Optional.ofNullable(saved) }
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetEventId = UUID.randomUUID(), decision = "event_created"))
        Mockito.`when`(commands.inputHash(anyLike(emptyList()))).thenAnswer { hashedFields = it.getArgument(0); "h" }
        // The keyed path runs its final write (the last argument) like the real executor's last transaction.
        Mockito.`when`(
            commands.executeEvent(anyLike(IdempotencyKey(UUID.randomUUID().toString())), anyLike(adminId),
                anyLike(AdminActionOperation.EVENT_CREATE), anyArg(), anyLike(""), anyLike {}, anyLike {}, anyLike { null },
                anyLike { _, _ -> }, anyLike {}, anyLike { _, _ -> AdminActionResult(null, null, null) }),
        ).thenAnswer {
            it.getArgument<(UUID, String?) -> AdminActionResult>(10).invoke(UUID.randomUUID(), null)
        }
        val aiProperties = Mockito.mock(AiApiProperties::class.java)
        val photos = Mockito.mock(PhotoRepository::class.java)
        service = AdminEventService(
            events, photos, decisions,
            EventDtoMapper(Mockito.mock(StorageService::class.java), StorageProperties(), photos,
                Mockito.mock(com.quickpitik.repository.EventPhotographerRepository::class.java)),
            covers,
            EventFaceExpiry(events, Mockito.mock(FaceBibProvider::class.java), aiProperties, java.time.Clock.systemUTC(), testTransactionTemplate()),
            aiProperties,
            Mockito.mock(PhotographerSettingsRepository::class.java),
            Mockito.mock(UserRepository::class.java),
            testTransactionTemplate(),
            commands,
            RegionsService(com.fasterxml.jackson.module.kotlin.jacksonObjectMapper()).apply { load() },
            RaceProperties(),
            raceEvents,
            Mockito.mock(com.quickpitik.repository.EventPhotographerRepository::class.java),
            Mockito.mock(com.quickpitik.service.events.RacePriceService::class.java),
        )
    }

    private fun request(eventType: String? = null) = CreateAdminEventRequest(
        title = "Cebu City Marathon 2026",
        date = "2026-12-06",
        location = "IT Park, Cebu City",
        pricePerPhoto = BigDecimal("125"),
        organizerName = "Cebu Runners",
        eventType = eventType,
    )

    private val key get() = IdempotencyKey(UUID.randomUUID().toString())

    @Test
    fun `typed admin create sets name key and admin approval under the race lock`() {
        for (keyed in listOf(false, true)) {
            Mockito.clearInvocations(events)
            saved = null

            val dto = service.create(adminId, request("ROAD_RACE"), key = if (keyed) key else null)

            val race = saved!!
            assertEquals(RaceType.ROAD_RACE, race.eventType, "keyed=$keyed")
            assertEquals("cebu city marathon", race.nameKey)
            assertNotNull(race.adminApprovedAt, "an admin-created race is proven at birth")
            assertNull(race.createdBy)
            assertNull(race.submittedBy)
            assertEquals("ROAD_RACE", dto.eventType)
            assertEquals(true, dto.adminApproved)
            val order = Mockito.inOrder(events)
            order.verify(events).lockRaceCreation()
            order.verify(events).save(anyArg<Event>())
        }
    }

    @Test
    fun `untyped admin create is unchanged`() {
        for (keyed in listOf(false, true)) {
            saved = null

            val dto = service.create(adminId, request(), key = if (keyed) key else null)

            val event = saved!!
            assertNull(event.eventType)
            assertNull(event.nameKey)
            assertNull(event.adminApprovedAt)
            assertNull(dto.eventType)
            assertEquals(false, dto.adminApproved)
        }
        Mockito.verify(events, Mockito.never()).lockRaceCreation()
    }

    @Test
    fun `keyed create fingerprint is unchanged without eventType`() {
        service.create(adminId, request(), key = key)
        val legacy = listOf("Cebu City Marathon 2026", "2026-12-06", "IT Park, Cebu City", "125.00", "Cebu Runners", "", "")
        assertEquals(legacy, hashedFields)

        service.create(adminId, request("TRAIL_RUN"), key = key)
        assertEquals(legacy + "TRAIL_RUN", hashedFields)
    }

    @Test
    fun `duplicate candidates pass the trimmed input to the matcher and refuse a blank name`() {
        val input = RaceInput("Cebu City Marathon", java.time.LocalDate.of(2026, 12, 6), "CEB", null)
        Mockito.`when`(raceEvents.findDuplicates(input, null)).thenReturn(emptyList())

        assertEquals(emptyList(), service.duplicateCandidates(" Cebu City Marathon ", "2026-12-06", " CEB ", " ", null))
        Mockito.verify(raceEvents).findDuplicates(input, null)

        val ex = assertFailsWith<ValidationException> { service.duplicateCandidates(" ", "2026-12-06", null, null, null) }
        assertEquals("name", ex.field)
    }

    @Test
    fun `an unknown race type is refused before any cover upload or write`() {
        for (code in listOf("OTHER", "MARATHON")) {
            val ex = assertFailsWith<ValidationException> {
                service.create(adminId, request(code), AdminEventService.CoverUpload(byteArrayOf(1), "image/jpeg"))
            }
            assertEquals("eventType", ex.field)
        }
        Mockito.verifyNoInteractions(covers)
        Mockito.verify(events, Mockito.never()).save(anyArg<Event>())
    }
}
