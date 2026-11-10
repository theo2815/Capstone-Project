package com.quickpitik.service.events

import com.quickpitik.config.IdentityProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.dto.photographer.CreateRaceRequest
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.RaceCheckReason
import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.moderation.ImageModerationResult
import com.quickpitik.service.moderation.ImageModerationService
import com.quickpitik.service.moderation.ImageModerationVerdict
import com.quickpitik.service.moderation.WordList
import com.quickpitik.service.moderation.WordListChecker
import com.quickpitik.service.photographer.PhotographerAccessGuard
import com.quickpitik.service.reference.RegionsService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZonedDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals

/** AA-R4: each creation check passes quietly or sends the race to an admin; nothing is refused. */
class RaceEventServiceChecksTest {
    private val me = UUID.randomUUID()
    private val manila = ZoneId.of("Asia/Manila")
    private val today = LocalDate.of(2026, 11, 9)
    private val coverKey = "events/covers/race.jpg"

    private lateinit var events: EventRepository
    private lateinit var coverage: EventPhotographerRepository
    private lateinit var covers: EventCoverService
    private lateinit var flags: SystemFlagService
    private lateinit var words: WordListChecker
    private lateinit var moderation: ImageModerationService
    private lateinit var publisher: ApplicationEventPublisher
    private lateinit var sellers: SellerDetailsRepository
    private lateinit var guard: PhotographerAccessGuard
    private lateinit var regions: RegionsService
    private var saved: Event? = null

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
        moderation = Mockito.mock(ImageModerationService::class.java)
        publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        sellers = Mockito.mock(SellerDetailsRepository::class.java)
        guard = Mockito.mock(PhotographerAccessGuard::class.java)
        regions = Mockito.mock(RegionsService::class.java)
        Mockito.`when`(guard.requireApproved(eqArg(me), Mockito.anyString())).thenReturn(PhotographerSettings(userId = me))
        Mockito.`when`(regions.isValidPair("region-7", "cebu")).thenReturn(true)
        val seller = Mockito.mock(SellerDetails::class.java)
        Mockito.`when`(seller.status).thenReturn(SellerDetailsStatus.APPROVED)
        Mockito.`when`(sellers.findById(me)).thenReturn(Optional.of(seller))
        Mockito.`when`(covers.upload(anyLike(UUID.randomUUID()), anyLike(ByteArray(0)), anyArg())).thenReturn(coverKey)
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(emptyList())
        Mockito.`when`(events.saveAndFlush(anyArg<Event>())).thenAnswer { it.getArgument<Event>(0).also { e -> saved = e } }
        Mockito.`when`(moderation.check(anyLike(ByteArray(0)))).thenReturn(ImageModerationResult(ImageModerationVerdict.CLEAN))
        Mockito.`when`(flags.raise(anyLike(FlagTargetKind.USER), anyLike(me), anyLike(SystemFlagReason.RACE_CHECK_MISS), anyArg(), anyArg()))
            .thenReturn(true)
        // Small stand-in lists: "run"/"marathon" are race words, "wedding" a non-race word, "damn" a bad word.
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

    private fun clockAt(hour: Int, minute: Int, day: LocalDate = today) =
        Clock.fixed(ZonedDateTime.of(day.atTime(hour, minute), manila).toInstant(), manila)

    private fun service(clock: Clock = clockAt(10, 0)) = RaceEventService(
        events, coverage, Mockito.mock(PhotographerSettingsRepository::class.java), sellers, guard,
        RaceProperties(selfServeEnabled = true), IdentityProperties(), PlatformProperties(), regions, covers,
        Mockito.mock(EventDtoMapper::class.java), flags, words,
        TransactionTemplate(Mockito.mock(PlatformTransactionManager::class.java).also {
            Mockito.`when`(it.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
        }),
        publisher, moderation, clock, Mockito.mock(EventFaceExpiry::class.java),
    )

    private fun request(
        title: String = "Cebu City Marathon",
        date: LocalDate = today.plusDays(20),
        organizer: String? = "Cebu Runners",
        description: String? = null,
    ) = CreateRaceRequest(title = title, date = date.toString(), location = "IT Park", regionCode = "region-7",
        provinceCode = "cebu", eventType = "ROAD_RACE", organizerName = organizer, description = description)

    private val cover = AdminEventService.CoverUpload(byteArrayOf(1, 2, 3), "image/jpeg")

    private fun assertPending(result: com.quickpitik.dto.photographer.RaceCreateResultDto, vararg reasons: RaceCheckReason) {
        assertEquals("pending_review", result.outcome)
        assertEquals(reasons.map { it.name }, result.checkReasons)
        assertEquals(EventStatus.DRAFT, saved!!.status)
        assertEquals(EventReviewStatus.PENDING, saved!!.reviewStatus)
        assertEquals(reasons.toList(), saved!!.checkReasons)
    }

    @Test
    fun `a wedding name goes pending with NON_RACE_WORD and a RACE_CHECK_MISS flag`() {
        val result = service().create(me, request(title = "Lim Wedding Fun Run"), null)

        assertPending(result, RaceCheckReason.NON_RACE_WORD)
        Mockito.verify(flags).raise(eqArg(FlagTargetKind.EVENT), eqArg(saved!!.id), eqArg(SystemFlagReason.RACE_CHECK_MISS),
            eqArg("NON_RACE_WORD"), Mockito.isNull())
        Mockito.verify(publisher).publishEvent(anyArg<Any>())
    }

    @Test
    fun `a bad word in the name, organizer or description goes pending with BAD_WORDS`() {
        for (req in listOf(request(title = "Damn City Marathon"), request(organizer = "Damn Runners"), request(description = "damn fast"))) {
            assertPending(service().create(me, req, null), RaceCheckReason.BAD_WORDS)
        }
    }

    @Test
    fun `a name with no race word goes pending with NO_RACE_WORD`() {
        assertPending(service().create(me, request(title = "Cebu Sunday Gathering"), null), RaceCheckReason.NO_RACE_WORD)
    }

    @Test
    fun `a race date 4 days past goes pending`() {
        assertPending(service().create(me, request(date = today.minusDays(4)), null), RaceCheckReason.DATE_OUT_OF_WINDOW)
        assertPending(service().create(me, request(date = today.plusDays(91)), null), RaceCheckReason.DATE_OUT_OF_WINDOW)
        assertEquals("created", service().create(me, request(date = today.minusDays(3)), null).outcome)
        assertEquals("created", service().create(me, request(date = today.plusDays(90)), null).outcome)
    }

    @Test
    fun `the fourth race of a Manila day goes pending with DAILY_CAP`() {
        val lateEvening = OffsetDateTime.of(today.atStartOfDay(), manila.rules.getOffset(today.atStartOfDay()))
        val nextMidnight = lateEvening.plusDays(1)
        Mockito.`when`(events.countRacesSubmittedSince(me, lateEvening)).thenReturn(3)
        Mockito.`when`(events.countRacesSubmittedSince(me, nextMidnight)).thenReturn(0)

        assertPending(service(clockAt(23, 59)).create(me, request(), null), RaceCheckReason.DAILY_CAP)
        // Two minutes later it is a new Manila day: the count starts again.
        assertEquals("created", service(clockAt(0, 1, today.plusDays(1))).create(me, request(), null).outcome)
    }

    @Test
    fun `AI off cover goes pending with COVER_UNCHECKED, never refused`() {
        Mockito.`when`(moderation.check(anyLike(ByteArray(0)))).thenReturn(ImageModerationResult(ImageModerationVerdict.UNCHECKED))

        assertPending(service().create(me, request(), cover), RaceCheckReason.COVER_UNCHECKED)
        assertEquals(coverKey, saved!!.coverS3Key, "the admin needs the cover to decide")
    }

    @Test
    fun `a flagged cover goes pending with COVER_FLAGGED`() {
        Mockito.`when`(moderation.check(anyLike(ByteArray(0)))).thenReturn(ImageModerationResult(ImageModerationVerdict.FLAGGED))

        assertPending(service().create(me, request(), cover), RaceCheckReason.COVER_FLAGGED)
    }

    @Test
    fun `races past the daily cap skip the cover check and raise one user flag`() {
        Mockito.`when`(events.countRacesSubmittedSince(eqArg(me), anyLike(OffsetDateTime.now()))).thenReturn(3, 3, 4, 4)

        service().create(me, request(), cover)
        Mockito.verify(flags).raise(eqArg(FlagTargetKind.USER), eqArg(me), eqArg(SystemFlagReason.RACE_CHECK_MISS),
            anyArg(), Mockito.isNull())
        Mockito.verify(publisher, Mockito.times(1)).publishEvent(anyArg<Any>())

        // The fifth: the user flag is already open, so a note is appended and no admin is pinged again.
        Mockito.`when`(flags.raise(anyLike(FlagTargetKind.USER), anyLike(me), anyLike(SystemFlagReason.RACE_CHECK_MISS), anyArg(), anyArg()))
            .thenReturn(false)
        assertPending(service().create(me, request(title = "Cebu Night Run"), cover), RaceCheckReason.DAILY_CAP)

        Mockito.verify(flags).appendNote(eqArg(FlagTargetKind.USER), eqArg(me), eqArg(SystemFlagReason.RACE_CHECK_MISS), Mockito.anyString())
        Mockito.verify(flags, Mockito.never()).raise(eqArg(FlagTargetKind.EVENT), anyLike(me), anyLike(SystemFlagReason.RACE_CHECK_MISS),
            anyArg(), anyArg())
        Mockito.verify(publisher, Mockito.times(1)).publishEvent(anyArg<Any>())
        Mockito.verifyNoInteractions(moderation)
        Mockito.verify(covers, Mockito.times(2)).upload(anyLike(UUID.randomUUID()), anyLike(ByteArray(0)), anyArg())
    }

    @Test
    fun `a race that tips over the cap inside the lock keeps its cover result and also gets DAILY_CAP`() {
        Mockito.`when`(events.countRacesSubmittedSince(eqArg(me), anyLike(OffsetDateTime.now()))).thenReturn(2, 3)
        Mockito.`when`(moderation.check(anyLike(ByteArray(0)))).thenReturn(ImageModerationResult(ImageModerationVerdict.UNCHECKED))

        // Reasons come in RaceCheckReason order.
        assertPending(service().create(me, request(), cover), RaceCheckReason.DAILY_CAP, RaceCheckReason.COVER_UNCHECKED)
    }

    @Test
    fun `a check-miss race gives the submitter a coverage row`() {
        service().create(me, request(title = "Lim Wedding Fun Run"), null)

        Mockito.verify(coverage).insertCoverageIfAbsent(saved!!.id, me, false)
    }

    @Test
    fun `a clean race with a clean cover goes live with no flag`() {
        val result = service().create(me, request(), cover)

        assertEquals("created", result.outcome)
        assertEquals(emptyList(), result.checkReasons)
        assertEquals(EventStatus.ACTIVE, saved!!.status)
        Mockito.verifyNoInteractions(flags, publisher)
    }
}
