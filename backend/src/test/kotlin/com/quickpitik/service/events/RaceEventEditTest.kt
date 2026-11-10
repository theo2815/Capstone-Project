package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IdentityProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.dto.photographer.UpdateRaceRequest
import com.quickpitik.dto.photographer.detailDto
import com.quickpitik.dto.photographer.summaryDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.RaceCheckReason
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.moderation.ImageModerationResult
import com.quickpitik.service.moderation.ImageModerationService
import com.quickpitik.service.moderation.ImageModerationVerdict
import com.quickpitik.service.moderation.WordList
import com.quickpitik.service.moderation.WordListChecker
import com.quickpitik.service.photographer.PhotographerAccessGuard
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.support.testTransactionTemplate
import org.hibernate.exception.ConstraintViolationException
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import java.sql.SQLException
import java.time.Clock
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/** AA-R5: the submitter edits their race until another photographer uploads; a failing edit is held. */
class RaceEventEditTest {
    private val me = UUID.randomUUID()
    private val other = UUID.randomUUID()
    private val manila = ZoneId.of("Asia/Manila")
    private val today = LocalDate.of(2026, 11, 9)
    private val clock = Clock.fixed(ZonedDateTime.of(today.atTime(10, 0), manila).toInstant(), manila)
    private val newCoverKey = "events/covers/new.jpg"

    private lateinit var events: EventRepository
    private lateinit var coverage: EventPhotographerRepository
    private lateinit var covers: EventCoverService
    private lateinit var flags: SystemFlagService
    private lateinit var words: WordListChecker
    private lateinit var publisher: ApplicationEventPublisher
    private lateinit var guard: PhotographerAccessGuard

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T { Mockito.eq(value); return value }
    private fun <T> anyLike(dummy: T): T { Mockito.any<T>(); return dummy }

    @BeforeEach
    fun setUp() {
        events = Mockito.mock(EventRepository::class.java)
        coverage = Mockito.mock(EventPhotographerRepository::class.java)
        covers = Mockito.mock(EventCoverService::class.java)
        flags = Mockito.mock(SystemFlagService::class.java)
        words = Mockito.mock(WordListChecker::class.java)
        publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        guard = Mockito.mock(PhotographerAccessGuard::class.java)
        Mockito.`when`(guard.requireApproved(eqArg(me), Mockito.anyString())).thenReturn(PhotographerSettings(userId = me))
        Mockito.`when`(covers.upload(anyLike(UUID.randomUUID()), anyLike(ByteArray(0)), anyArg())).thenReturn(newCoverKey)
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(emptyList())
        Mockito.`when`(events.saveAndFlush(anyArg<Event>())).thenAnswer { it.getArgument<Event>(0) }
        Mockito.`when`(flags.raise(anyLike(FlagTargetKind.EVENT), anyLike(me), anyLike(SystemFlagReason.RACE_EDIT_HELD), anyArg(), anyArg()))
            .thenReturn(true)
        // "run"/"marathon" are race words, "wedding" a non-race word, "damn" a bad word.
        Mockito.`when`(words.hits(Mockito.anyString(), anyLike(emptySet()))).thenAnswer {
            val text = it.getArgument<String>(0).lowercase()
            val lists = it.getArgument<Set<WordList>>(1)
            buildList {
                if (WordList.RACE_WORDS in lists) addAll(listOf("run", "marathon").filter { w -> w in text })
                if (WordList.NON_RACE_WORDS in lists && "wedding" in text) add("wedding")
                if (WordList.BAD_WORDS in lists && "damn" in text) add("damn")
            }
        }
    }

    private fun service(races: RaceProperties = RaceProperties(selfServeEnabled = true)) = RaceEventService(
        events, coverage, Mockito.mock(PhotographerSettingsRepository::class.java), Mockito.mock(SellerDetailsRepository::class.java),
        guard, races, IdentityProperties(), PlatformProperties(),
        Mockito.mock(RegionsService::class.java).also { Mockito.`when`(it.isValidPair("region-7", "cebu")).thenReturn(true) },
        covers, Mockito.mock(EventDtoMapper::class.java), flags, words, testTransactionTemplate(), publisher,
        Mockito.mock(ImageModerationService::class.java).also {
            Mockito.`when`(it.check(anyLike(ByteArray(0)))).thenReturn(ImageModerationResult(ImageModerationVerdict.CLEAN))
        },
        clock,
        EventFaceExpiry(events, Mockito.mock(FaceBibProvider::class.java), AiApiProperties(), clock, testTransactionTemplate()),
    )

    private fun race(
        review: EventReviewStatus = EventReviewStatus.APPROVED,
        date: LocalDate = today.plusDays(20),
        otherUploaded: Boolean = false,
    ): Event {
        val event = Event(
            slug = "cebu-city-marathon", name = "Cebu City Marathon", date = date, location = "IT Park",
            status = if (review == EventReviewStatus.PENDING) EventStatus.DRAFT else EventStatus.ACTIVE,
            reviewStatus = review, regionCode = "region-7", provinceCode = "cebu",
        ).apply {
            eventType = RaceType.ROAD_RACE
            submittedBy = me
            nameKey = "cebu city marathon"
            coverS3Key = "events/covers/live.jpg"
        }
        Mockito.`when`(events.findRaceForSubmitterForUpdate(event.id, me)).thenReturn(event)
        Mockito.`when`(coverage.existsOtherUploader(event.id, me)).thenReturn(otherUploaded)
        return event
    }

    private val cover = AdminEventService.CoverUpload(byteArrayOf(1, 2, 3), "image/jpeg")

    @Test
    fun `submitter edit applies before anyone else uploads`() {
        val event = race()

        val dto = service().update(me, event.id, UpdateRaceRequest(title = "Cebu South Marathon"), null)

        assertEquals("Cebu South Marathon", event.name)
        assertEquals("cebu marathon south", event.nameKey)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertEquals("Cebu South Marathon", dto.name)
        Mockito.verifyNoInteractions(flags)
    }

    @Test
    fun `edit after another photographer uploaded answers RACE_EDIT_LOCKED`() {
        val event = race(otherUploaded = true)

        val ex = assertFailsWith<ApiException> { service().update(me, event.id, UpdateRaceRequest(title = "Cebu South Marathon"), cover) }

        assertEquals(ErrorCodes.RACE_EDIT_LOCKED, ex.code)
        assertEquals(403, ex.status.value())
        assertEquals("Cebu City Marathon", event.name)
        Mockito.verifyNoInteractions(covers)
    }

    @Test
    fun `a failing rename is held and the live name stays`() {
        val event = race()

        val dto = service().update(me, event.id, UpdateRaceRequest(title = "Lim Wedding Run"), null)

        assertEquals("Cebu City Marathon", event.name)
        assertEquals("cebu city marathon", event.nameKey)
        assertEquals(EventReviewStatus.CHANGE_PENDING, event.reviewStatus)
        assertEquals(listOf(RaceCheckReason.NON_RACE_WORD), event.checkReasons)
        val held = assertNotNull(event.pendingChange)
        assertEquals("race_edit", held["kind"])
        assertEquals(mapOf("title" to "Lim Wedding Run"), held["fields"])
        assertEquals(me.toString(), held["requestedBy"])
        assertEquals("change_pending", dto.reviewStatus)
        Mockito.verify(flags).raise(eqArg(FlagTargetKind.EVENT), eqArg(event.id), eqArg(SystemFlagReason.RACE_EDIT_HELD),
            anyLike("note"), Mockito.isNull())
    }

    @Test
    fun `an unchanged old date never trips the window`() {
        val event = race(date = today.minusDays(10))

        service().update(me, event.id, UpdateRaceRequest(title = "Cebu City Night Marathon", date = today.minusDays(10).toString()), null)

        assertEquals("Cebu City Night Marathon", event.name)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.checkReasons)
    }

    @Test
    fun `a date moved past face retention is refused`() {
        val event = race()

        val ex = assertFailsWith<ValidationException> {
            service().update(me, event.id, UpdateRaceRequest(date = today.minusMonths(7).toString()), cover)
        }

        assertEquals("date", ex.field)
        assertEquals(today.plusDays(20), event.date)
        Mockito.verifyNoInteractions(covers)
    }

    @Test
    fun `a passing edit clears the held request`() {
        val event = race().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            checkReasons = listOf(RaceCheckReason.NON_RACE_WORD)
            pendingChange = mapOf("kind" to "race_edit", "fields" to mapOf("title" to "Lim Wedding Run"),
                "coverS3Key" to "events/covers/held.jpg", "requestedBy" to me.toString(), "requestedAt" to "2026-11-08T10:00:00Z")
        }

        service().update(me, event.id, UpdateRaceRequest(title = "Cebu South Marathon"), cover)

        assertEquals("Cebu South Marathon", event.name)
        assertEquals(newCoverKey, event.coverS3Key)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertNull(event.checkReasons)
        Mockito.verify(covers).delete("events/covers/held.jpg")
        Mockito.verify(covers).delete("events/covers/live.jpg")
        Mockito.verify(covers, Mockito.never()).delete(newCoverKey)
    }

    @Test
    fun `a pending race edit applies at once and refreshes its reasons`() {
        val event = race(review = EventReviewStatus.PENDING).apply {
            name = "Lim Wedding Run"
            checkReasons = listOf(RaceCheckReason.COVER_UNCHECKED, RaceCheckReason.NON_RACE_WORD)
        }

        service().update(me, event.id, UpdateRaceRequest(title = "Lim Family Fun Run"), null)

        assertEquals("Lim Family Fun Run", event.name)
        assertEquals(EventReviewStatus.PENDING, event.reviewStatus)
        assertEquals(listOf(RaceCheckReason.COVER_UNCHECKED), event.checkReasons)
        assertNull(event.pendingChange)
    }

    @Test
    fun `a unique violation on an edit answers RACE_DUPLICATE and drops the new cover`() {
        val event = race()
        Mockito.`when`(events.saveAndFlush(anyArg<Event>())).thenThrow(
            DataIntegrityViolationException("dup", ConstraintViolationException("dup", SQLException("dup"), "uq_events_race_date_name")),
        )

        val ex = assertFailsWith<ConflictException> {
            service().update(me, event.id, UpdateRaceRequest(title = "Cebu South Marathon"), cover)
        }

        assertEquals(ErrorCodes.RACE_DUPLICATE, ex.code)
        Mockito.verify(covers).delete(newCoverKey)
    }

    @Test
    fun `a rename onto another race on the same day is held with POSSIBLE_DUPLICATE`() {
        val event = race()
        val other = Event(slug = "cebu-south-marathon", name = "Cebu South Marathon", date = event.date, location = "SRP",
            status = EventStatus.ACTIVE, provinceCode = "cebu").apply { eventType = RaceType.ROAD_RACE }
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(listOf(event, other))

        service().update(me, event.id, UpdateRaceRequest(title = "Cebu South Marathon"), null)

        assertEquals("Cebu City Marathon", event.name)
        assertEquals(EventReviewStatus.CHANGE_PENDING, event.reviewStatus)
        assertEquals(listOf(RaceCheckReason.POSSIBLE_DUPLICATE), event.checkReasons)
        Mockito.verify(events).lockRaceCreation()
        val note = org.mockito.ArgumentCaptor.forClass(String::class.java)
        Mockito.verify(flags).raise(eqArg(FlagTargetKind.EVENT), eqArg(event.id), eqArg(SystemFlagReason.RACE_EDIT_HELD),
            note.capture(), Mockito.isNull())
        kotlin.test.assertContains(note.value, other.id.toString())
    }

    @Test
    fun `switch off answers SELF_SERVE_RACES_DISABLED`() {
        val event = race()

        val ex = assertFailsWith<ApiException> {
            service(RaceProperties(selfServeEnabled = false)).update(me, event.id, UpdateRaceRequest(title = "Cebu South Marathon"), null)
        }

        assertEquals(ErrorCodes.SELF_SERVE_RACES_DISABLED, ex.code)
    }

    @Test
    fun `a held race edit is shown only to the submitter`() {
        val event = race().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("kind" to "race_edit", "fields" to mapOf("title" to "Lim Wedding Run"),
                "coverS3Key" to "events/covers/held.jpg", "requestedBy" to me.toString())
        }
        fun ep(viewer: UUID) = EventPhotographer(id = EventPhotographerId(event.id, viewer))

        assertNull(detailDto(event, ep(other), null, viewerId = other).pendingChange)
        assertNull(summaryDto(event, ep(other), null, viewerId = other).pendingChange)
        assertEquals(event.pendingChange, detailDto(event, ep(me), null, viewerId = me).pendingChange)
        assertEquals(event.pendingChange, summaryDto(event, ep(me), null, viewerId = me).pendingChange)
    }
}
