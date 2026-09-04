package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.UserSelfie
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotoAlertRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserSelfieRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.LocalDate
import java.time.ZoneId
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Opt-in service: a runner with no selfie can't be matched, so registering is
// rejected rather than creating a dead alert; explicit selfieIds are IDOR-safe;
// re-registering updates the row rather than duplicating it.
class EventPhotoAlertServiceTest {

    private lateinit var eventRepository: EventRepository
    private lateinit var userSelfieRepository: UserSelfieRepository
    private lateinit var alertRepository: EventPhotoAlertRepository
    private lateinit var service: EventPhotoAlertService

    private val userId = UUID.randomUUID()
    private val slug = "cebu-marathon"
    private val event = Event(
        slug = slug,
        name = "Cebu Marathon",
        date = LocalDate.now(ZoneId.of("Asia/Manila")),
        location = "Cebu City, Cebu",
        status = EventStatus.ACTIVE,
    )

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        userSelfieRepository = Mockito.mock(UserSelfieRepository::class.java)
        alertRepository = Mockito.mock(EventPhotoAlertRepository::class.java)
        service = EventPhotoAlertService(eventRepository, userSelfieRepository, alertRepository)

        Mockito.`when`(eventRepository.findBySlugAndDeletedAtIsNull(slug)).thenReturn(event)
        Mockito.`when`(alertRepository.save(anyArg<EventPhotoAlert>())).thenAnswer { it.arguments[0] }
    }

    @Test
    fun `register with no selfie is rejected`() {
        Mockito.`when`(userSelfieRepository.findFirstByUserIdAndIsPrimaryTrue(userId)).thenReturn(null)
        Mockito.`when`(userSelfieRepository.findFirstByUserIdOrderByUploadedAtDesc(userId)).thenReturn(null)

        val ex = assertFailsWith<ValidationException> { service.register(userId, slug, null) }

        assertEquals(ErrorCodes.SELFIE_REQUIRED, ex.code)
        Mockito.verify(alertRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `register with no selfieId uses the primary selfie`() {
        val selfie = selfie()
        Mockito.`when`(userSelfieRepository.findFirstByUserIdAndIsPrimaryTrue(userId)).thenReturn(selfie)
        Mockito.`when`(alertRepository.findByEventIdAndUserId(event.id, userId)).thenReturn(null)

        val dto = service.register(userId, slug, null)

        assertTrue(dto.registered)
        assertEquals(selfie.id, dto.selfieId)
    }

    @Test
    fun `register with a selfieId the runner does not own is rejected`() {
        val someoneElses = UUID.randomUUID()
        Mockito.`when`(userSelfieRepository.findByIdAndUserId(someoneElses, userId)).thenReturn(null)

        val ex = assertFailsWith<NotFoundException> {
            service.register(userId, slug, someoneElses.toString())
        }

        assertEquals(ErrorCodes.SELFIE_NOT_FOUND, ex.code)
    }

    @Test
    fun `re-registering updates the existing row rather than duplicating`() {
        val selfie = selfie()
        val existing = EventPhotoAlert(eventId = event.id, userId = userId, selfieId = UUID.randomUUID())
        Mockito.`when`(userSelfieRepository.findFirstByUserIdAndIsPrimaryTrue(userId)).thenReturn(selfie)
        Mockito.`when`(alertRepository.findByEventIdAndUserId(event.id, userId)).thenReturn(existing)

        val dto = service.register(userId, slug, null)

        assertEquals(selfie.id, existing.selfieId) // mutated in place
        assertEquals(selfie.id, dto.selfieId)
        Mockito.verify(alertRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `register rejects an event after its upload window`() {
        event.date = LocalDate.now(ZoneId.of("Asia/Manila")).minusDays(4)

        val ex = assertFailsWith<ApiException> { service.register(userId, slug, null) }

        assertEquals(ErrorCodes.EVENT_NOT_UPLOADABLE, ex.code)
        Mockito.verify(alertRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `register accepts a completed event still inside its upload window`() {
        event.status = EventStatus.COMPLETED
        val selfie = selfie()
        Mockito.`when`(userSelfieRepository.findFirstByUserIdAndIsPrimaryTrue(userId)).thenReturn(selfie)

        val dto = service.register(userId, slug, null)

        assertTrue(dto.registered)
    }

    @Test
    fun `register rejects an archived event`() {
        event.status = EventStatus.ARCHIVED

        val ex = assertFailsWith<ApiException> { service.register(userId, slug, null) }

        assertEquals(ErrorCodes.EVENT_NOT_UPLOADABLE, ex.code)
        Mockito.verify(alertRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `optOut reports whether a row was removed`() {
        Mockito.`when`(alertRepository.deleteByEventIdAndUserId(event.id, userId)).thenReturn(1L)
        assertTrue(service.optOut(userId, slug))

        Mockito.`when`(alertRepository.deleteByEventIdAndUserId(event.id, userId)).thenReturn(0L)
        assertFalse(service.optOut(userId, slug))
    }

    private fun selfie() = UserSelfie(
        id = UUID.randomUUID(),
        userId = userId,
        s3Key = "selfies/$userId/x.jpg",
        isPrimary = true,
    )

    private fun <T> anyArg(): T = Mockito.any()
}
