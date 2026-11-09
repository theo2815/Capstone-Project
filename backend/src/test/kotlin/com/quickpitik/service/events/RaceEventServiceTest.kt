package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.IdentityProperties
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.dto.photographer.CreateRaceRequest
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.RaceCheckReason
import com.quickpitik.entity.RaceType
import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.moderation.WordListChecker
import com.quickpitik.service.photographer.PhotographerAccessGuard
import com.quickpitik.service.reference.RegionsService
import org.hibernate.exception.ConstraintViolationException
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.sql.SQLException
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RaceEventServiceTest {
    private val me = UUID.randomUUID()
    private val raceDate = LocalDate.of(2026, 12, 6)
    private val coverKey = "events/covers/race.jpg"

    private lateinit var events: EventRepository
    private lateinit var coverage: EventPhotographerRepository
    private lateinit var settingsRepo: PhotographerSettingsRepository
    private lateinit var sellers: SellerDetailsRepository
    private lateinit var guard: PhotographerAccessGuard
    private lateinit var regions: RegionsService
    private lateinit var covers: EventCoverService
    private lateinit var mapper: EventDtoMapper
    private lateinit var flags: SystemFlagService
    private lateinit var words: WordListChecker
    private lateinit var publisher: ApplicationEventPublisher
    private var saved: Event? = null

    private fun <T> anyArg(): T = Mockito.any()
    // Mocked Kotlin classes null-check their parameters, so these return a real value after registering the matcher.
    private fun <T> eqArg(value: T): T { Mockito.eq(value); return value }
    private fun <T> anyLike(dummy: T): T { Mockito.any<T>(); return dummy }

    @BeforeEach
    fun setUp() {
        events = Mockito.mock(EventRepository::class.java)
        coverage = Mockito.mock(EventPhotographerRepository::class.java)
        settingsRepo = Mockito.mock(PhotographerSettingsRepository::class.java)
        sellers = Mockito.mock(SellerDetailsRepository::class.java)
        guard = Mockito.mock(PhotographerAccessGuard::class.java)
        regions = Mockito.mock(RegionsService::class.java)
        covers = Mockito.mock(EventCoverService::class.java)
        mapper = Mockito.mock(EventDtoMapper::class.java)
        flags = Mockito.mock(SystemFlagService::class.java)
        words = Mockito.mock(WordListChecker::class.java)
        publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        val settings = PhotographerSettings(userId = me).apply { metadataPublicByDefault = true }
        Mockito.`when`(guard.requireApproved(eqArg(me), Mockito.anyString())).thenReturn(settings)
        Mockito.`when`(settingsRepo.findById(me)).thenReturn(Optional.of(settings))
        Mockito.`when`(regions.isValidPair("07", "CEB")).thenReturn(true)
        seller(SellerDetailsStatus.AUTO_APPROVED)
        Mockito.`when`(covers.upload(anyLike(UUID.randomUUID()), anyLike(ByteArray(0)), anyArg())).thenReturn(coverKey)
        // Every test race name passes the AA-R4 checks: it has a race word and nothing else is hit.
        Mockito.`when`(words.hits(Mockito.anyString(), anyLike(emptySet()))).thenAnswer {
            if (com.quickpitik.service.moderation.WordList.RACE_WORDS in it.getArgument<Set<*>>(1)) listOf("marathon") else emptyList()
        }
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(emptyList())
        Mockito.`when`(events.findAliasKeys(anyArg())).thenReturn(emptyList())
        Mockito.`when`(events.saveAndFlush(anyArg<Event>())).thenAnswer { it.getArgument<Event>(0).also { e -> saved = e } }
    }

    private fun seller(status: SellerDetailsStatus) {
        val row = Mockito.mock(SellerDetails::class.java)
        Mockito.`when`(row.status).thenReturn(status)
        Mockito.`when`(sellers.findById(me)).thenReturn(Optional.of(row))
        Mockito.`when`(sellers.existsByUserIdAndStatusIn(anyArg(), anyArg())).thenAnswer {
            it.getArgument<UUID>(0) == me && status in it.getArgument<Collection<SellerDetailsStatus>>(1)
        }
    }

    private fun service(
        races: RaceProperties = RaceProperties(selfServeEnabled = true),
        identity: IdentityProperties = IdentityProperties(),
    ) = RaceEventService(
        events, coverage, settingsRepo, sellers, guard, races, identity, PlatformProperties(), regions, covers, mapper,
        flags, words,
        TransactionTemplate(Mockito.mock(PlatformTransactionManager::class.java).also {
            Mockito.`when`(it.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
        }),
        publisher,
        Mockito.mock(com.quickpitik.service.moderation.ImageModerationService::class.java).also {
            Mockito.`when`(it.check(anyLike(ByteArray(0))))
                .thenReturn(com.quickpitik.service.moderation.ImageModerationResult(com.quickpitik.service.moderation.ImageModerationVerdict.CLEAN))
        },
        // Fixed "today", so the December race dates stay inside the AA-R4 date window.
        java.time.Clock.fixed(java.time.Instant.parse("2026-11-09T02:00:00Z"), java.time.ZoneOffset.UTC),
    )

    private fun request(title: String = "Cebu City Marathon 2026", confirmDifferentFrom: UUID? = null) = CreateRaceRequest(
        title = title,
        date = raceDate.toString(),
        location = "IT Park, Cebu City",
        regionCode = "07",
        provinceCode = "CEB",
        eventType = "ROAD_RACE",
        organizerName = "Cebu Runners",
        confirmDifferentFrom = confirmDifferentFrom,
    )

    private val cover = AdminEventService.CoverUpload(byteArrayOf(1, 2, 3), "image/jpeg")

    private fun existing(name: String) = Event(
        slug = "existing-${UUID.randomUUID()}",
        name = name,
        date = raceDate,
        location = "Cebu City",
        status = EventStatus.ACTIVE,
        provinceCode = "CEB",
    ).apply { eventType = RaceType.ROAD_RACE }

    // ─── create ───

    @Test
    fun `no match creates an active approved shared race with coverage and name key`() {
        val result = service().create(me, request(), cover)

        assertEquals("created", result.outcome)
        val race = saved!!
        assertEquals(EventStatus.ACTIVE, race.status)
        assertEquals(EventReviewStatus.APPROVED, race.reviewStatus)
        assertEquals(EventPricingMode.PAID, race.pricingMode)
        assertEquals(WatermarkPolicy.PLATFORM, race.watermarkPolicy)
        assertEquals(EventVisibility.PUBLIC, race.visibility)
        assertEquals(BigDecimal("125"), race.pricePerPhoto)
        assertNull(race.createdBy)
        assertEquals(me, race.submittedBy)
        assertEquals(RaceType.ROAD_RACE, race.eventType)
        assertEquals("cebu city marathon", race.nameKey)
        assertEquals("07", race.regionCode)
        assertEquals(coverKey, race.coverS3Key)
        assertNull(race.adminApprovedAt, "a self-serve race is not admin approved")
        Mockito.verify(coverage).insertCoverageIfAbsent(race.id, me, true)
        Mockito.verify(events).lockRaceCreation()
        Mockito.verify(covers, Mockito.never()).delete(Mockito.anyString())
        Mockito.verifyNoInteractions(flags)
    }

    @Test
    fun `strong match returns the match and stores nothing`() {
        val twin = existing("CEBU CITY MARATHON")
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(listOf(twin))

        val result = service().create(me, request(), cover)

        assertEquals("match", result.outcome)
        assertEquals(listOf(twin.id), result.candidates.map { it.id })
        assertEquals("strong", result.candidates.single().strength)
        assertNull(result.event)
        Mockito.verify(events, Mockito.never()).saveAndFlush(anyArg<Event>())
        Mockito.verifyNoInteractions(covers)
    }

    @Test
    fun `medium match returns candidates`() {
        val near = existing("Cebu Marathon")
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(listOf(near))

        val result = service().create(me, request(), cover)

        assertEquals("candidates", result.outcome)
        assertEquals("medium", result.candidates.single().strength)
        Mockito.verify(events, Mockito.never()).saveAndFlush(anyArg<Event>())
        Mockito.verifyNoInteractions(covers)
    }

    @Test
    fun `mine is different stores a pending draft pointing at the candidate and raises a flag`() {
        val near = existing("Cebu Marathon")
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(listOf(near))

        val result = service().create(me, request(confirmDifferentFrom = near.id), cover)

        assertEquals("pending_review", result.outcome)
        assertEquals(listOf("POSSIBLE_DUPLICATE"), result.checkReasons)
        val race = saved!!
        assertEquals(EventStatus.DRAFT, race.status)
        assertEquals(EventReviewStatus.PENDING, race.reviewStatus)
        assertEquals(near.id, race.possibleDuplicateOf)
        assertEquals(listOf(RaceCheckReason.POSSIBLE_DUPLICATE), race.checkReasons)
        val note = ArgumentCaptor.forClass(String::class.java)
        Mockito.verify(flags).raise(eqArg(FlagTargetKind.EVENT), eqArg(race.id), eqArg(SystemFlagReason.POSSIBLE_DUPLICATE),
            note.capture(), Mockito.isNull())
        assertTrue(near.id.toString() in note.value && "Cebu Marathon" in note.value && "medium" in note.value, note.value)
        Mockito.verify(publisher).publishEvent(anyArg<Any>())
    }

    @Test
    fun `mine is different on a strong match stores a pending draft with POSSIBLE_DUPLICATE`() {
        val twin = existing("CEBU CITY MARATHON")
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(listOf(twin))

        val result = service().create(me, request(confirmDifferentFrom = twin.id), cover)

        assertEquals("pending_review", result.outcome)
        assertEquals(twin.id, saved!!.possibleDuplicateOf)
        assertEquals(EventReviewStatus.PENDING, saved!!.reviewStatus)
        Mockito.verify(flags).raise(eqArg(FlagTargetKind.EVENT), eqArg(saved!!.id), eqArg(SystemFlagReason.POSSIBLE_DUPLICATE),
            anyArg(), Mockito.isNull())
    }

    @Test
    fun `every stored race gives the submitter a coverage row`() {
        service().create(me, request(), null)
        val created = saved!!
        val near = existing("Cebu Marathon")
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(listOf(near))
        service().create(me, request(confirmDifferentFrom = near.id), null)
        val declared = saved!!

        Mockito.verify(coverage).insertCoverageIfAbsent(created.id, me, true)
        Mockito.verify(coverage).insertCoverageIfAbsent(declared.id, me, true)
    }

    @Test
    fun `a race found inside the lock deletes the uploaded cover`() {
        val twin = existing("CEBU CITY MARATHON")
        // Nothing at the unlocked pre-check; another photographer's race is there once the lock is held.
        Mockito.`when`(events.findRaceCandidates(anyArg(), anyArg())).thenReturn(emptyList(), listOf(twin))

        val result = service().create(me, request(), cover)

        assertEquals("match", result.outcome)
        Mockito.verify(events, Mockito.never()).saveAndFlush(anyArg<Event>())
        Mockito.verify(covers).delete(coverKey)
    }

    @Test
    fun `a unique violation at flush deletes the uploaded cover and answers RACE_DUPLICATE`() {
        Mockito.`when`(events.saveAndFlush(anyArg<Event>())).thenThrow(
            DataIntegrityViolationException("dup", ConstraintViolationException("dup", SQLException("dup"), "uq_events_race_date_name")),
        )

        val error = assertFailsWith<ConflictException> { service().create(me, request(), cover) }

        assertEquals(ErrorCodes.RACE_DUPLICATE, error.code)
        Mockito.verify(covers).delete(coverKey)
    }

    @Test
    fun `another constraint violation at flush is not reported as a duplicate`() {
        Mockito.`when`(events.saveAndFlush(anyArg<Event>())).thenThrow(
            DataIntegrityViolationException("other", ConstraintViolationException("other", SQLException("other"), "events_slug_key")),
        )

        assertFailsWith<DataIntegrityViolationException> { service().create(me, request(), cover) }
        Mockito.verify(covers).delete(coverKey)
    }

    @Test
    fun `switch off answers SELF_SERVE_RACES_DISABLED`() {
        val off = service(races = RaceProperties(selfServeEnabled = false))

        for (error in listOf(
            assertFailsWith<ApiException> { off.create(me, request(), cover) },
            assertFailsWith<ApiException> { off.join(me, UUID.randomUUID(), null) },
        )) {
            assertEquals(HttpStatus.FORBIDDEN, error.status)
            assertEquals(ErrorCodes.SELF_SERVE_RACES_DISABLED, error.code)
        }
        Mockito.verifyNoInteractions(covers, guard)
    }

    @Test
    fun `identity gate on refuses an unverified seller with SELLER_DETAILS_REQUIRED`() {
        seller(SellerDetailsStatus.MANUAL_REVIEW)
        val gateOn = service(identity = IdentityProperties(gateEnabled = true))
        val race = existing("Cebu City Marathon")
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))

        for (error in listOf(
            assertFailsWith<ApiException> { gateOn.create(me, request(), cover) },
            assertFailsWith<ApiException> { gateOn.join(me, race.id, null) },
        )) {
            assertEquals(ErrorCodes.SELLER_DETAILS_REQUIRED, error.code)
        }
        Mockito.verifyNoInteractions(covers)
    }

    @Test
    fun `create refuses a MANUAL_REVIEW seller with the identity gate off`() {
        seller(SellerDetailsStatus.MANUAL_REVIEW)
        val race = existing("Cebu City Marathon")
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))

        val error = assertFailsWith<ApiException> { service().create(me, request(), cover) }

        assertEquals(ErrorCodes.SELLER_DETAILS_REQUIRED, error.code)
        Mockito.verifyNoInteractions(covers)
        assertEquals("joined", service().join(me, race.id, null).outcome, "join follows the normal upload rule")
    }

    // ─── join ───

    @Test
    fun `join is idempotent and skips a bad-word alias`() {
        val race = existing("Cebu City Marathon").apply { nameKey = "cebu city marathon" }
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))
        Mockito.`when`(words.hits(eqArg("Cebu City Marathon damn"), anyLike(emptySet()))).thenReturn(listOf("damn"))

        service().join(me, race.id, "Cebu City Marathon damn")
        service().join(me, race.id, "Cebu City Marathon 2026") // same key as the race: no alias
        service().join(me, race.id, "Sugbo City Marathon")

        Mockito.verify(coverage, Mockito.times(3)).insertCoverageIfAbsent(race.id, me, true)
        Mockito.verify(events).insertAlias(race.id, "city marathon sugbo", me)
        Mockito.verify(events, Mockito.times(1)).insertAlias(anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `join refuses a photographer-owned, rejected or deleted event with EVENT_NOT_JOINABLE`() {
        val owned = existing("Mine").apply { createdBy = UUID.randomUUID() }
        val rejected = existing("No").apply { reviewStatus = EventReviewStatus.REJECTED }
        val deleted = existing("Gone").apply { deletedAt = java.time.OffsetDateTime.now() }
        for (event in listOf(owned, rejected, deleted)) {
            Mockito.`when`(events.findById(event.id)).thenReturn(Optional.of(event))
            val error = assertFailsWith<ApiException> { service().join(me, event.id, null) }
            assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, error.status)
            assertEquals(ErrorCodes.EVENT_NOT_JOINABLE, error.code)
        }
        Mockito.verify(coverage, Mockito.never()).insertCoverageIfAbsent(anyArg(), anyArg(), Mockito.anyBoolean())
    }

    @Test
    fun `a race reads ownedByMe false and submittedByMe true`() {
        val result = service().create(me, request(), null)

        assertFalse(result.event!!.ownedByMe)
        assertTrue(result.event!!.submittedByMe)
        assertEquals("ROAD_RACE", result.event!!.eventType)
    }
}
