package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.entity.AdminActionOperation
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.RaceType
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.events.EventFaceExpiry
import com.quickpitik.service.events.RaceEventService
import com.quickpitik.service.events.RacePriceService
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.mockito.Mockito
import org.springframework.dao.DataIntegrityViolationException
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class AdminEventServiceRacePriceTest {
    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> anyLike(value: T): T { Mockito.any<T>(); return value }

    private val admin = UUID.randomUUID()
    private val events = Mockito.mock(EventRepository::class.java)
    private val photos = Mockito.mock(PhotoRepository::class.java)
    private val coverage = Mockito.mock(EventPhotographerRepository::class.java)
    private val prices = Mockito.mock(RacePriceService::class.java)

    private fun raceAt(price: String) = Event(slug = "race", name = "Race", date = LocalDate.of(2026, 12, 6), location = "Cebu",
        status = EventStatus.ACTIVE, pricePerPhoto = BigDecimal(price)).apply { eventType = RaceType.ROAD_RACE }

    private fun serviceFor(race: Event): AdminEventService {
        val decisions = Mockito.mock(AdminDecisionLogService::class.java)
        val commands = Mockito.mock(AdminActionCommandService::class.java)
        Mockito.`when`(events.findByIdForReview(race.id)).thenReturn(race)
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = admin, targetEventId = race.id, decision = "event_updated"))
        Mockito.`when`(commands.inputHash(anyLike(emptyList()))).thenReturn("hash")
        Mockito.`when`(commands.executeEvent(anyLike(IdempotencyKey(UUID.randomUUID().toString())), anyLike(admin),
            anyLike(AdminActionOperation.EVENT_UPDATE), anyArg(), anyLike(""), anyLike {}, anyLike {}, anyLike { null },
            anyLike { _, _ -> }, anyLike {}, anyLike { _, _ -> AdminActionResult(null, null, null) }))
            .thenAnswer { it.getArgument<(UUID, String?) -> AdminActionResult>(10).invoke(race.id, null) }
        return AdminEventService(events, photos, decisions,
            EventDtoMapper(Mockito.mock(StorageService::class.java), StorageProperties(), photos,
                Mockito.mock(com.quickpitik.repository.EventPhotographerRepository::class.java)),
            Mockito.mock(EventCoverService::class.java), Mockito.mock(EventFaceExpiry::class.java), AiApiProperties(),
            Mockito.mock(PhotographerSettingsRepository::class.java), Mockito.mock(UserRepository::class.java),
            testTransactionTemplate(), commands, Mockito.mock(RegionsService::class.java), RaceProperties(),
            Mockito.mock(RaceEventService::class.java), coverage, prices)
    }

    @ParameterizedTest
    @ValueSource(booleans = [false, true])
    fun `price PATCH on a race calls applyFloor before changing the event`(keyed: Boolean) {
        val race = raceAt("125.00")
        Mockito.doAnswer {
            assertEquals(BigDecimal("125.00"), race.pricePerPhoto, "native re-pricing must not auto-flush the new floor first")
            null
        }.`when`(prices).applyFloor(race, BigDecimal("150.00"), admin)
        val service = serviceFor(race)

        service.update(admin, race.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("150")),
            key = if (keyed) IdempotencyKey(UUID.randomUUID().toString()) else null)

        val order = Mockito.inOrder(photos, coverage, events, prices)
        order.verify(photos).lockIdsByEventIdForUpdate(race.id)
        order.verify(coverage).lockByEventIdForUpdate(race.id)
        order.verify(events).findByIdForReview(race.id)
        order.verify(prices).applyFloor(race, BigDecimal("150.00"), admin)
        assertEquals(BigDecimal("150.00"), race.pricePerPhoto)
    }

    // AF-10 (R-11): the native re-pricing flushes the rename first, so a name clash surfaces inside the price step.
    private fun clash(index: String) = DataIntegrityViolationException("dup",
        org.hibernate.exception.ConstraintViolationException("dup", java.sql.SQLException("dup"), index))

    @ParameterizedTest
    @ValueSource(booleans = [false, true])
    fun `a rename that collides during the price step answers RACE_DUPLICATE`(keyed: Boolean) {
        val race = raceAt("125.00")
        Mockito.`when`(prices.applyFloor(race, BigDecimal("150.00"), admin)).thenThrow(clash("uq_events_race_date_name"))
        val service = serviceFor(race)

        val ex = assertFailsWith<ConflictException> {
            service.update(admin, race.id, UpdateAdminEventRequest(title = "Cebu City Marathon", pricePerPhoto = BigDecimal("150")),
                key = if (keyed) IdempotencyKey(UUID.randomUUID().toString()) else null)
        }

        assertEquals(ErrorCodes.RACE_DUPLICATE, ex.code)
    }

    @Test
    fun `another integrity error in the price step is rethrown`() {
        val race = raceAt("125.00")
        val other = clash("some_other_constraint")
        Mockito.`when`(prices.applyFloor(race, BigDecimal("150.00"), admin)).thenThrow(other)
        val service = serviceFor(race)

        val ex = assertFailsWith<DataIntegrityViolationException> {
            service.update(admin, race.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("150")))
        }

        assertEquals(other, ex)
    }
}
