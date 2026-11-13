package com.quickpitik.service.admin

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.WatermarkPolicy
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.events.EventCoverService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.aop.framework.ProxyFactory
import org.springframework.transaction.annotation.AnnotationTransactionAttributeSource
import org.springframework.transaction.interceptor.TransactionInterceptor
import org.springframework.transaction.support.TransactionSynchronization
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Admin review of photographer-owned events (V46). One approve + one reject
// endpoint serve both queue states: an initial submission (PENDING, DRAFT)
// and a pricing edit request on a live event (CHANGE_PENDING, ACTIVE).
class AdminEventServiceReviewTest {
    private val adminId = UUID.randomUUID()
    private val photographerId = UUID.randomUUID()
    private lateinit var eventRepository: EventRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var decisions: AdminDecisionLogService
    private lateinit var service: AdminEventService
    private val pushed = mutableListOf<PhotographerMessageKind>()
    private val transactions = testTransactionTemplate()
    private lateinit var covers: EventCoverService
    private lateinit var provider: FaceBibProvider
    private lateinit var aiProperties: AiApiProperties
    private val raceEvents: com.quickpitik.service.events.RaceEventService = Mockito.mock(com.quickpitik.service.events.RaceEventService::class.java)
    private val settingsRepo: PhotographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
    private val users: UserRepository = Mockito.mock(UserRepository::class.java)
    private val storage: StorageService = Mockito.mock(StorageService::class.java)

    @BeforeEach
    fun setUp() {
        eventRepository = Mockito.mock(EventRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        decisions = Mockito.mock(AdminDecisionLogService::class.java)
        Mockito.`when`(eventRepository.save(anyArg())).thenAnswer { it.arguments[0] }
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetEventId = UUID.randomUUID(), decision = "x"))
        Mockito.`when`(decisions.pushMessage(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())).thenAnswer { inv ->
            pushed += inv.getArgument<PhotographerMessageKind>(1)
            null
        }
        covers = Mockito.mock(EventCoverService::class.java)
        provider = Mockito.mock(FaceBibProvider::class.java)
        aiProperties = Mockito.mock(AiApiProperties::class.java)
        val target = AdminEventService(
            eventRepository,
            photoRepository,
            decisions,
            EventDtoMapper(storage, StorageProperties(), photoRepository,
                Mockito.mock(com.quickpitik.repository.EventPhotographerRepository::class.java)),
            covers,
            com.quickpitik.service.events.EventFaceExpiry(eventRepository, provider, aiProperties, java.time.Clock.systemUTC(), transactions),
            aiProperties,
            settingsRepo,
            users,
            transactions,
            Mockito.mock(AdminActionCommandService::class.java),
            com.quickpitik.service.reference.RegionsService(com.fasterxml.jackson.module.kotlin.jacksonObjectMapper()).apply { load() },
            com.quickpitik.config.RaceProperties(),
            raceEvents,
            Mockito.mock(com.quickpitik.repository.EventPhotographerRepository::class.java),
            com.quickpitik.service.events.RacePriceService(eventRepository,
                Mockito.mock(com.quickpitik.repository.EventPhotographerRepository::class.java), photoRepository,
                Mockito.mock(com.quickpitik.repository.RacePriceChangeRepository::class.java),
                Mockito.mock(com.quickpitik.service.photographer.PhotographerAccessGuard::class.java),
                com.quickpitik.config.RaceProperties(), com.quickpitik.config.PlatformProperties(), java.time.Clock.systemUTC(), decisions),
        )
        service = ProxyFactory(target).apply {
            addAdvice(TransactionInterceptor().apply {
                transactionManager = transactions.transactionManager
                transactionAttributeSource = AnnotationTransactionAttributeSource()
            })
        }.proxy as AdminEventService
    }

    @Test
    fun `approving a submission opens the event and tells the photographer`() {
        val event = submitted()

        service.approve(adminId, event.id)

        assertEquals(EventStatus.ACTIVE, event.status)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertEquals(adminId, event.reviewedBy)
        assertNotNull(event.reviewedAt)
        assertEquals(listOf(PhotographerMessageKind.EVENT_APPROVED), pushed)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_approved"), anyArg(), anyArg())
    }

    @Test
    fun `approving a pricing change applies it, re-prices photos and re-renders previews`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "free", "pricePerPhoto" to "0.00", "watermarkPolicy" to "own")
        }

        service.approve(adminId, event.id)

        assertEquals(EventPricingMode.FREE, event.pricingMode)
        assertEquals(0, event.pricePerPhoto.signum())
        assertEquals(WatermarkPolicy.OWN, event.watermarkPolicy)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertEquals(EventStatus.ACTIVE, event.status)
        Mockito.verify(photoRepository).updatePriceByEventId(eqArg(event.id), anyArg())
        Mockito.verify(photoRepository).resetForRewatermark(event.id)
        assertEquals(listOf(PhotographerMessageKind.EVENT_CHANGE_APPROVED), pushed)
    }

    @Test
    fun `approving a submission takes no photo locks`() {
        val event = submitted()

        service.approve(adminId, event.id)

        Mockito.verify(photoRepository, Mockito.never()).lockIdsByEventIdForUpdate(anyArg())
    }

    @Test
    fun `approving a pricing change locks its photos before the event row`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "paid", "pricePerPhoto" to "200.00", "watermarkPolicy" to "platform")
        }

        service.approve(adminId, event.id)

        val order = Mockito.inOrder(photoRepository, eventRepository)
        order.verify(photoRepository).lockIdsByEventIdForUpdate(event.id)
        order.verify(eventRepository).findByIdForReview(event.id)
    }

    @Test
    fun `a pricing change that appears after the unlocked read is refused, not applied without photo locks`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "paid", "pricePerPhoto" to "200.00", "watermarkPolicy" to "platform")
        }
        Mockito.`when`(eventRepository.findReviewStatus(event.id)).thenReturn(EventReviewStatus.PENDING)

        assertFailsWith<ConflictException> { service.approve(adminId, event.id) }

        assertEquals(EventReviewStatus.CHANGE_PENDING, event.reviewStatus)
        Mockito.verify(photoRepository, Mockito.never()).updatePriceByEventId(anyArg(), anyArg())
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `a price-only change does not re-render previews`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "paid", "pricePerPhoto" to "200.00", "watermarkPolicy" to "platform")
        }

        service.approve(adminId, event.id)

        assertEquals(0, BigDecimal("200.00").compareTo(event.pricePerPhoto))
        Mockito.verify(photoRepository).updatePriceByEventId(eqArg(event.id), anyArg())
        Mockito.verify(photoRepository, Mockito.never()).resetForRewatermark(anyArg())
    }

    @Test
    fun `approve and reject refuse an event that is not in the queue`() {
        val event = live()

        assertFailsWith<ConflictException> { service.approve(adminId, event.id) }
        assertFailsWith<ConflictException> { service.reject(adminId, event.id, "no") }
        assertEquals(EventStatus.ACTIVE, event.status)
    }

    @Test
    fun `rejecting a submission keeps it a draft with the reason`() {
        val event = submitted()

        service.reject(adminId, event.id, "Price too high for a fun run")

        assertEquals(EventStatus.DRAFT, event.status)
        assertEquals(EventReviewStatus.REJECTED, event.reviewStatus)
        assertEquals("Price too high for a fun run", event.reviewNote)
        assertEquals(listOf(PhotographerMessageKind.EVENT_REJECTED), pushed)
    }

    @Test
    fun `rejecting a pricing change leaves the live event untouched`() {
        val event = live().apply {
            reviewStatus = EventReviewStatus.CHANGE_PENDING
            pendingChange = mapOf("pricingMode" to "free", "pricePerPhoto" to "0.00", "watermarkPolicy" to "none")
        }

        service.reject(adminId, event.id, "Keep it paid")

        assertEquals(EventPricingMode.PAID, event.pricingMode)
        assertEquals(0, BigDecimal("150.00").compareTo(event.pricePerPhoto))
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertEquals("Keep it paid", event.reviewNote)
        assertEquals(EventStatus.ACTIVE, event.status)
        assertEquals(listOf(PhotographerMessageKind.EVENT_CHANGE_REJECTED), pushed)
        Mockito.verify(photoRepository, Mockito.never()).resetForRewatermark(anyArg())
    }

    @Test
    fun `approve and reject read the row under a lock`() {
        val event = submitted()

        service.approve(adminId, event.id)

        Mockito.verify(eventRepository).findByIdForReview(event.id)
        Mockito.verify(eventRepository, Mockito.never()).findById(anyArg())
    }

    @Test
    fun `an admin price edit on a free event is refused`() {
        val event = live().apply {
            pricingMode = EventPricingMode.FREE
            pricePerPhoto = BigDecimal.ZERO
            watermarkPolicy = WatermarkPolicy.OWN
        }

        assertFailsWith<ValidationException> {
            service.update(adminId, event.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("100")))
        }
        assertEquals(0, event.pricePerPhoto.signum())
        Mockito.verify(photoRepository, Mockito.never()).updatePriceByEventId(anyArg(), anyArg())
    }

    // Admin ₱0 parity (2026-09-05): a PAID event at ₱0 could be created but
    // never sold, so zero is refused unless the event is FREE.
    @Test
    fun `an admin price edit to zero is refused on a paid event and allowed on a free one`() {
        val paid = live()
        val ex = assertFailsWith<ValidationException> {
            service.update(adminId, paid.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal.ZERO))
        }
        assertEquals("pricePerPhoto", ex.field)
        assertEquals(BigDecimal("150.00"), paid.pricePerPhoto)

        val free = live().apply {
            pricingMode = EventPricingMode.FREE
            pricePerPhoto = BigDecimal.ZERO
            watermarkPolicy = WatermarkPolicy.OWN
        }
        service.update(adminId, free.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal.ZERO))
        assertEquals(0, free.pricePerPhoto.signum())
    }

    @Test
    fun `the review queue lists only pending submissions and change requests`() {
        val queue = listOf(submitted(), live().apply { reviewStatus = EventReviewStatus.CHANGE_PENDING })
        Mockito.`when`(eventRepository.findByReviewStatusInAndDeletedAtIsNullOrderByCreatedAtAsc(anyArg()))
            .thenReturn(queue)

        val page = service.list(stateFilter = null, review = "queue", params = PaginationParams.of(0, 20))

        assertEquals(2, page.total)
        Mockito.verify(eventRepository, Mockito.never()).pageForAdmin(anyArg(), anyArg(), anyArg(), anyArg())
    }

    // AA-R7: the queue tells the admin why a race waits, who submitted it and which race it may repeat.
    private fun candidate(name: String, strength: String) = com.quickpitik.dto.photographer.RaceCandidateDto(
        UUID.randomUUID(), "cebu-marathon", name, LocalDate.of(2026, 10, 10), "Cebu City", "07", "CEB", "", "ROAD_RACE",
        null, "approved", strength,
    )

    private fun queueOf(vararg rows: Event) {
        Mockito.`when`(eventRepository.findByReviewStatusInAndDeletedAtIsNullOrderByCreatedAtAsc(anyArg()))
            .thenReturn(rows.toList())
    }

    @Test
    fun `queue shows possible duplicate and reasons for a pending race`() {
        val race = pendingRace().apply { possibleDuplicateOf = UUID.randomUUID() }
        val twin = candidate("Cebu City Marathon", "strong")
        queueOf(race)
        Mockito.`when`(raceEvents.duplicatesOf(listOf(race))).thenReturn(mapOf(race.id to twin))
        Mockito.`when`(settingsRepo.findAllById(setOf(photographerId)))
            .thenReturn(listOf(com.quickpitik.entity.PhotographerSettings(userId = photographerId).apply { handle = "juan" }))
        val user = Mockito.mock(com.quickpitik.entity.User::class.java)
        Mockito.`when`(user.id).thenReturn(photographerId)
        Mockito.`when`(user.name).thenReturn("Juan Dela Cruz")
        Mockito.`when`(users.findAllById(setOf(photographerId))).thenReturn(listOf(user))

        val row = service.list(stateFilter = null, review = "queue", params = PaginationParams.of(0, 20)).items.single()

        assertEquals("ROAD_RACE", row.eventType)
        assertEquals(listOf("POSSIBLE_DUPLICATE"), row.checkReasons)
        assertEquals(twin, row.possibleDuplicateOf)
        assertEquals("juan", row.submittedByHandle)
        assertEquals("Juan Dela Cruz", row.submittedByName)
        assertNull(row.createdByHandle, "a race has no owner")
        assertNull(row.lookalike)
        assertFalse(row.adminApproved)
        assertNull(row.hiddenAt)
    }

    // AF-10 (R-5): a held edit's new cover is only a key in pendingChange; the admin row signs it like the live cover.
    @Test
    fun `a held race edit with a new cover carries a signed link`() {
        val race = heldRace()
        queueOf(race)
        Mockito.`when`(storage.presignedGetUrl("events/covers/held.jpg", StorageProperties().presignedTtl.cover))
            .thenReturn("https://signed.test/held")

        val row = service.list(stateFilter = null, review = "queue", params = PaginationParams.of(0, 20)).items.single()

        assertEquals("https://signed.test/held", row.heldCoverUrl)
    }

    @Test
    fun `a row with no held cover has no link`() {
        queueOf(live().apply { reviewStatus = EventReviewStatus.CHANGE_PENDING; coverS3Key = "events/covers/live.jpg" })

        val row = service.list(stateFilter = null, review = "queue", params = PaginationParams.of(0, 20)).items.single()

        assertNull(row.heldCoverUrl)
    }

    @Test
    fun `queue shows the lookalike race for a pending Other event`() {
        val other = submitted().apply { name = "Cebu City Marathon" }
        val race = candidate("Cebu City Marathon", "medium")
        queueOf(other)
        Mockito.`when`(raceEvents.lookalikes(listOf(other))).thenReturn(mapOf(other.id to race))

        val row = service.list(stateFilter = null, review = "queue", params = PaginationParams.of(0, 20)).items.single()

        assertEquals(race, row.lookalike)
        assertNull(row.eventType)
        assertNull(row.possibleDuplicateOf)
        assertEquals(emptyList(), row.checkReasons)
    }

    @Test
    fun `create uploads before opening the event and audit transaction`() {
        var uploadedId: UUID? = null
        Mockito.`when`(covers.upload(anyArg(), anyArg(), anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            Mockito.verify(eventRepository, Mockito.never()).save(anyArg())
            uploadedId = it.getArgument(0)
            "new-cover"
        }
        assertWritesTransactional()

        val result = service.create(adminId, createRequest(), cover())

        assertEquals(uploadedId, result.id)
        Mockito.verify(eventRepository).save(anyArg())
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(result.id), eqArg("event_created"), anyArg(), anyArg())
    }

    @Test
    fun `update uploads outside the transaction and deletes the old cover only after commit`() {
        val event = live().apply { coverS3Key = "old-cover" }
        var committed = false
        var deleted = false
        Mockito.`when`(covers.upload(eqArg(event.id), anyArg(), anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            "new-cover"
        }
        assertWritesTransactional { committed = true }
        Mockito.doAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            assertTrue(committed)
            deleted = true
            null
        }.`when`(covers).delete("old-cover")

        val result = service.update(
            adminId, event.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("200")), cover(), removeCover = true,
        )

        assertTrue(deleted)
        assertEquals("new-cover", event.coverS3Key)
        assertEquals(BigDecimal("200.00"), result.pricePerPhoto)
        assertEquals("set", (result.adminOverrides.single()["before"] as Map<*, *>)["cover"])
        assertEquals("set", (result.adminOverrides.single()["after"] as Map<*, *>)["cover"])
        Mockito.verify(photoRepository).updatePriceByEventId(event.id, BigDecimal("200.00"))
    }

    @Test
    fun `removing a cover waits for commit and records the override`() {
        val event = live().apply { coverS3Key = "old-cover" }
        var committed = false
        assertWritesTransactional { committed = true }
        Mockito.doAnswer {
            assertTrue(committed)
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            null
        }.`when`(covers).delete("old-cover")

        val result = service.update(adminId, event.id, UpdateAdminEventRequest(), removeCover = true)

        assertNull(event.coverS3Key)
        assertEquals("none", (result.adminOverrides.single()["after"] as Map<*, *>)["cover"])
        Mockito.verify(covers).delete("old-cover")
    }

    @Test
    fun `failed cover upload does not persist a new event or audit`() {
        Mockito.`when`(covers.upload(anyArg(), anyArg(), anyArg())).thenThrow(IllegalArgumentException("bad cover"))

        assertFailsWith<IllegalArgumentException> { service.create(adminId, createRequest(), cover()) }

        Mockito.verify(eventRepository, Mockito.never()).save(anyArg())
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `failed update audit preserves the old cover object`() {
        val event = live().apply { coverS3Key = "old-cover" }
        Mockito.`when`(covers.upload(eqArg(event.id), anyArg(), anyArg())).thenReturn("new-cover")
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenThrow(IllegalStateException("audit unavailable"))

        assertFailsWith<IllegalStateException> {
            service.update(adminId, event.id, UpdateAdminEventRequest(), cover())
        }

        Mockito.verify(covers, Mockito.never()).delete("old-cover")
    }

    @Test
    fun `delete commits its audit before best effort provider erasure`() {
        val event = live()
        var committed = false
        var providerWasTransactional: Boolean? = null
        var providerSawCommit = false
        Mockito.`when`(aiProperties.enabled).thenReturn(true)
        assertWritesTransactional { committed = true }
        Mockito.doAnswer {
            providerWasTransactional = TransactionSynchronizationManager.isActualTransactionActive()
            providerSawCommit = committed
            throw IllegalStateException("provider unavailable")
        }.`when`(provider).deleteFacesByEvent(event.id)

        assertTrue(service.delete(adminId, event.id).removed)

        assertEquals(false, providerWasTransactional)
        assertTrue(providerSawCommit)
        assertNotNull(event.deletedAt)
        assertNull(event.facesDeletedAt) // failed erasure: the daily face-expiry sweep retries it
        assertFalse(service.delete(adminId, event.id).removed)
        assertFalse(service.delete(adminId, UUID.randomUUID()).removed)
        Mockito.verify(provider).deleteFacesByEvent(event.id)
    }

    @Test
    fun `delete stamps faces_deleted_at when the provider erasure succeeds`() {
        val event = live()
        Mockito.`when`(aiProperties.enabled).thenReturn(true)

        assertTrue(service.delete(adminId, event.id).removed)

        Mockito.verify(provider).deleteFacesByEvent(event.id)
        assertNotNull(event.facesDeletedAt)
    }

    @Test
    fun `failed delete audit never erases provider faces`() {
        val event = live()
        Mockito.`when`(aiProperties.enabled).thenReturn(true)
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg()))
            .thenThrow(IllegalStateException("audit unavailable"))

        assertFailsWith<IllegalStateException> { service.delete(adminId, event.id) }

        Mockito.verifyNoInteractions(provider)
    }

    @Test
    fun `invalid updates are rejected before uploading a cover`() {
        val paid = live()
        assertFailsWith<NotFoundException> {
            service.update(adminId, UUID.randomUUID(), UpdateAdminEventRequest(), cover())
        }
        assertFailsWith<ValidationException> {
            service.update(adminId, paid.id, UpdateAdminEventRequest(date = "invalid"), cover())
        }
        assertFailsWith<ValidationException> {
            service.update(adminId, paid.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal.ZERO), cover())
        }
        val free = live().apply { pricingMode = EventPricingMode.FREE }
        assertFailsWith<ValidationException> {
            service.update(adminId, free.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("100")), cover())
        }
        Mockito.verifyNoInteractions(covers)
    }

    @Test
    fun `delete skips provider erasure while AI is disabled`() {
        val event = live()

        assertTrue(service.delete(adminId, event.id).removed)

        Mockito.verifyNoInteractions(provider)
    }

    @Test
    fun `an edit after deletion is not found and cannot reprice photos`() {
        val event = live().apply { deletedAt = java.time.OffsetDateTime.now() }
        assertFailsWith<NotFoundException> {
            service.update(adminId, event.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("200")))
        }
        Mockito.verifyNoInteractions(decisions)
        Mockito.verify(photoRepository, Mockito.never()).updatePriceByEventId(anyArg(), anyArg())
    }

    @Test
    fun `a repeated unchanged text edit has only one override and audit`() {
        val event = live()
        val request = UpdateAdminEventRequest(title = "New name")
        service.update(adminId, event.id, request)
        service.update(adminId, event.id, request)
        assertEquals(1, event.adminOverrides.size)
        Mockito.verify(decisions, Mockito.times(1)).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_updated"), anyArg(), anyArg())
    }

    @Test
    fun `a deletion during cover upload is rechecked before applying any edit`() {
        val event = live().apply { coverS3Key = "old-cover" }
        Mockito.`when`(covers.upload(eqArg(event.id), anyArg(), anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            event.deletedAt = java.time.OffsetDateTime.now()
            "new-cover"
        }
        assertFailsWith<NotFoundException> {
            service.update(adminId, event.id, UpdateAdminEventRequest(title = "After delete"), cover())
        }
        assertEquals("Fun Run", event.name)
        assertEquals("old-cover", event.coverS3Key)
        Mockito.verifyNoInteractions(decisions)
        Mockito.verify(covers, Mockito.never()).delete("old-cover")
    }

    private fun assertWritesTransactional(onCommit: () -> Unit = {}) {
        Mockito.`when`(eventRepository.save(anyArg())).thenAnswer {
            assertTrue(TransactionSynchronizationManager.isActualTransactionActive())
            TransactionSynchronizationManager.registerSynchronization(object : TransactionSynchronization {
                override fun afterCommit() = onCommit()
            })
            it.arguments[0]
        }
        Mockito.`when`(decisions.logEventDecision(anyArg(), anyArg(), anyArg(), anyArg(), anyArg())).thenAnswer {
            assertTrue(TransactionSynchronizationManager.isActualTransactionActive())
            AdminDecisionLog(adminId = adminId, targetEventId = it.getArgument(1), decision = it.getArgument(2))
        }
    }

    @Test
    fun `invalid location pairs reject legacy and keyed writes before cover or claim`() {
        val event = submitted()
        for ((region, province) in listOf("region-7" to null, null to "cebu", "" to "",
            "region-7" to "metro-manila", "r".repeat(21) to "cebu", "region-7" to "p".repeat(41))) {
            for (key in listOf(null, com.quickpitik.common.IdempotencyKey(UUID.randomUUID().toString()))) {
                val createError = assertFailsWith<ValidationException> {
                    service.create(adminId, createRequest().copy(regionCode = region, provinceCode = province), cover(), key)
                }
                assertEquals("INVALID_REGION", createError.code)
                val updateError = assertFailsWith<ValidationException> {
                    service.update(adminId, event.id, UpdateAdminEventRequest(regionCode = region, provinceCode = province), cover(), key = key)
                }
                assertEquals("INVALID_REGION", updateError.code)
            }
        }
        Mockito.verifyNoInteractions(covers, decisions)
    }

    // ── "Show on /events" (V89) ──────────────────────────────────────────
    @Test
    fun `listing a public photographer event shows it on events and logs the decision`() {
        val event = live()
        val revision = event.reviewRevision

        val row = service.setListing(adminId, event.id, listed = true)

        assertTrue(event.listedOnEvents)
        assertTrue(row.listedOnEvents)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertEquals(revision, event.reviewRevision)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_listed"), anyArg(), anyArg())
        Mockito.verify(eventRepository).save(event)
    }

    @Test
    fun `listing an admin event is refused because it is always listed`() {
        val event = live().apply { createdBy = null }

        val ex = assertFailsWith<ValidationException> { service.setListing(adminId, event.id, listed = true) }

        assertEquals("Admin events are always listed on /events.", ex.message)
        assertFalse(event.listedOnEvents)
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `listing a private event is refused until the photographer makes it public`() {
        val event = live().apply { visibility = EventVisibility.UNLISTED }

        val ex = assertFailsWith<ValidationException> { service.setListing(adminId, event.id, listed = true) }

        assertEquals("This event is Private. The photographer must make it Public first.", ex.message)
        assertFalse(event.listedOnEvents)
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `listing a draft event is refused until it is approved`() {
        val event = live().apply { status = EventStatus.DRAFT }

        val ex = assertFailsWith<ValidationException> { service.setListing(adminId, event.id, listed = true) }

        assertEquals("This event is still waiting for review. Approve it before showing it on /events.", ex.message)
        assertFalse(event.listedOnEvents)
        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `unlisting always works even on a private event`() {
        val event = live().apply { visibility = EventVisibility.UNLISTED; listedOnEvents = true }

        val row = service.setListing(adminId, event.id, listed = false)

        assertFalse(event.listedOnEvents)
        assertFalse(row.listedOnEvents)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_unlisted"), anyArg(), anyArg())
    }

    @Test
    fun `listing an unknown event is not found`() {
        assertFailsWith<NotFoundException> { service.setListing(adminId, UUID.randomUUID(), listed = true) }
    }

    @Test
    fun `setting the listing to its current value writes no second audit row`() {
        val event = live().apply { listedOnEvents = true }

        val row = service.setListing(adminId, event.id, listed = true)

        assertTrue(row.listedOnEvents)
        Mockito.verifyNoInteractions(decisions)
    }

    private fun createRequest() = CreateAdminEventRequest(
        title = "Fun Run", date = "2026-10-10", location = "Cebu", pricePerPhoto = BigDecimal("150"),
    )

    private fun cover() = AdminEventService.CoverUpload(byteArrayOf(1), "image/jpeg")

    // AA-R5: a live shared race (createdBy NULL, submittedBy set) with a held race_edit, dated ahead so
    // the held date move stays inside face retention.
    private val raceDay: LocalDate = LocalDate.now().plusDays(30)

    private fun heldRace(): Event = event(EventStatus.ACTIVE, EventReviewStatus.CHANGE_PENDING).apply {
        Mockito.`when`(aiProperties.faceRetention).thenReturn(java.time.Period.ofMonths(6))
        date = raceDay
        createdBy = null
        submittedBy = photographerId
        eventType = com.quickpitik.entity.RaceType.ROAD_RACE
        name = "Cebu City Marathon"
        nameKey = "cebu city marathon"
        coverS3Key = "events/covers/live.jpg"
        checkReasons = listOf(com.quickpitik.entity.RaceCheckReason.NON_RACE_WORD)
        pendingChange = mapOf(
            "kind" to "race_edit",
            "fields" to mapOf("title" to "Cebu South Marathon", "date" to raceDay.plusDays(1).toString()),
            "coverS3Key" to "events/covers/held.jpg",
            "requestedBy" to photographerId.toString(),
            "requestedAt" to "2026-11-08T10:00:00Z",
        )
        Mockito.`when`(eventRepository.saveAndFlush(this)).thenReturn(this)
    }

    @Test
    fun `approving a held race edit applies fields and name key without touching prices`() {
        val event = heldRace()

        service.approve(adminId, event.id)

        assertEquals("Cebu South Marathon", event.name)
        assertEquals("cebu marathon south", event.nameKey)
        assertEquals(raceDay.plusDays(1), event.date)
        assertEquals("events/covers/held.jpg", event.coverS3Key)
        assertEquals(BigDecimal("150.00"), event.pricePerPhoto)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertNull(event.checkReasons)
        Mockito.verify(photoRepository, Mockito.never()).updatePriceByEventId(anyArg(), anyArg())
        Mockito.verify(photoRepository, Mockito.never()).resetForRewatermark(anyArg())
        Mockito.verify(covers).delete("events/covers/live.jpg")
        // The race's submitter hears about it (createdBy is NULL on a race).
        assertEquals(listOf(PhotographerMessageKind.EVENT_CHANGE_APPROVED), pushed)
        Mockito.verify(decisions).pushMessage(eqArg(photographerId), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())
    }

    @Test
    fun `rejecting a held race edit deletes the held cover`() {
        val event = heldRace()

        service.reject(adminId, event.id, "Not a race name")

        assertEquals("Cebu City Marathon", event.name)
        assertEquals("events/covers/live.jpg", event.coverS3Key)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNull(event.pendingChange)
        assertNull(event.checkReasons)
        Mockito.verify(covers).delete("events/covers/held.jpg")
        Mockito.verify(covers, Mockito.never()).delete("events/covers/live.jpg")
        assertEquals(listOf(PhotographerMessageKind.EVENT_CHANGE_REJECTED), pushed)
    }

    @Test
    fun `a unique violation on approving a held race edit answers RACE_DUPLICATE`() {
        val event = heldRace()
        Mockito.`when`(eventRepository.saveAndFlush(event)).thenThrow(
            org.springframework.dao.DataIntegrityViolationException("dup", org.hibernate.exception.ConstraintViolationException(
                "dup", java.sql.SQLException("dup"), "uq_events_race_date_name")),
        )

        val ex = assertFailsWith<ConflictException> { service.approve(adminId, event.id) }

        assertEquals(com.quickpitik.common.ErrorCodes.RACE_DUPLICATE, ex.code)
        Mockito.verify(covers, Mockito.never()).delete(anyArg())
    }

    @Test
    fun `an admin rename of a race recomputes its name key`() {
        val event = heldRace().apply { reviewStatus = EventReviewStatus.APPROVED; pendingChange = null }

        service.update(adminId, event.id, UpdateAdminEventRequest(title = "Cebu North Marathon"))

        assertEquals("Cebu North Marathon", event.name)
        assertEquals("cebu marathon north", event.nameKey)
        Mockito.verify(eventRepository).saveAndFlush(event)
    }

    // ── AA-R6: admin decisions on shared races ───────────────────────────
    private fun asRace(event: Event, name: String = "Cebu City Marathon"): Event = event.apply {
        createdBy = null
        submittedBy = photographerId
        eventType = com.quickpitik.entity.RaceType.ROAD_RACE
        this.name = name
        nameKey = name.lowercase()
    }

    private fun pendingRace(): Event = asRace(submitted()).apply {
        checkReasons = listOf(com.quickpitik.entity.RaceCheckReason.POSSIBLE_DUPLICATE)
    }

    @Test
    fun `approving a pending race marks it admin approved and messages the submitter`() {
        val event = pendingRace()

        service.approve(adminId, event.id)

        assertEquals(EventStatus.ACTIVE, event.status)
        assertEquals(EventReviewStatus.APPROVED, event.reviewStatus)
        assertNotNull(event.adminApprovedAt)
        assertNull(event.checkReasons)
        assertEquals(listOf(PhotographerMessageKind.EVENT_APPROVED), pushed)
        Mockito.verify(decisions).pushMessage(eqArg(photographerId), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())
    }

    @Test
    fun `rejecting a pending race clears its check reasons`() {
        val event = pendingRace()

        service.reject(adminId, event.id, "Not a real race")

        assertEquals(EventReviewStatus.REJECTED, event.reviewStatus)
        assertNull(event.checkReasons)
        assertNull(event.adminApprovedAt)
    }

    @Test
    fun `approving an untyped submission does not mark it admin approved`() {
        val event = submitted()

        service.approve(adminId, event.id)

        assertNull(event.adminApprovedAt)
    }

    @Test
    fun `hiding and relisting a shared race toggles hidden_at and relist proves it`() {
        val event = asRace(live())

        service.setListing(adminId, event.id, listed = false)

        assertNotNull(event.hiddenAt)
        assertNull(event.adminApprovedAt)
        assertFalse(event.listedOnEvents)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_hidden"), anyArg(), anyArg())

        service.setListing(adminId, event.id, listed = true)

        assertNull(event.hiddenAt)
        assertNotNull(event.adminApprovedAt)
        assertFalse(event.listedOnEvents)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_relisted"), anyArg(), anyArg())
    }

    @Test
    fun `hiding a race that is already hidden writes no second audit row`() {
        val event = asRace(live()).apply { hiddenAt = java.time.OffsetDateTime.now() }

        service.setListing(adminId, event.id, listed = false)

        Mockito.verifyNoInteractions(decisions)
    }

    @Test
    fun `same race joins the submitter to the target and removes the pending row`() {
        val target = asRace(live(), name = "Cebu Marathon")
        val event = pendingRace().apply {
            possibleDuplicateOf = target.id
            coverS3Key = "events/covers/pending.jpg"
        }
        Mockito.`when`(raceEvents.joinInternal(target.id, photographerId, event.name)).thenReturn(target)

        val row = service.sameRace(adminId, event.id)

        assertEquals(target.id, row.id)
        Mockito.verify(raceEvents).joinInternal(target.id, photographerId, "Cebu City Marathon")
        assertNotNull(event.deletedAt)
        assertEquals(EventReviewStatus.REJECTED, event.reviewStatus)
        assertEquals("Same race as Cebu Marathon", event.reviewNote)
        assertNull(event.checkReasons)
        Mockito.verify(decisions).logEventDecision(eqArg(adminId), eqArg(event.id), eqArg("event_same_race"), anyArg(), anyArg())
        assertEquals(listOf(PhotographerMessageKind.ADMIN_MESSAGE), pushed)
        Mockito.verify(decisions).pushMessage(eqArg(photographerId), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.isNull())
        Mockito.verify(covers).delete("events/covers/pending.jpg")
    }

    @Test
    fun `same race takes an explicit target over the stored duplicate`() {
        val target = asRace(live(), name = "Cebu Marathon")
        val event = pendingRace().apply { possibleDuplicateOf = UUID.randomUUID() }
        Mockito.`when`(raceEvents.joinInternal(target.id, photographerId, event.name)).thenReturn(target)

        service.sameRace(adminId, event.id, targetId = target.id)

        Mockito.verify(raceEvents).joinInternal(target.id, photographerId, "Cebu City Marathon")
    }

    @Test
    fun `same race refuses a row that is not a pending race or has no target`() {
        val untyped = submitted()
        val live = asRace(live())
        val noTarget = pendingRace()
        val self = pendingRace().apply { possibleDuplicateOf = id }

        assertFailsWith<ConflictException> { service.sameRace(adminId, untyped.id, targetId = UUID.randomUUID()) }
        assertFailsWith<ConflictException> { service.sameRace(adminId, live.id, targetId = UUID.randomUUID()) }
        assertFailsWith<ValidationException> { service.sameRace(adminId, noTarget.id) }
        assertFailsWith<ValidationException> { service.sameRace(adminId, self.id) }

        Mockito.verifyNoInteractions(raceEvents, decisions)
        assertNull(noTarget.deletedAt)
    }

    @Test
    fun `same race refuses a stale revision`() {
        val event = pendingRace().apply { possibleDuplicateOf = UUID.randomUUID() }

        assertFailsWith<com.quickpitik.exception.ApiException> {
            service.sameRace(adminId, event.id, expectedRevision = event.reviewRevision + 1)
        }

        Mockito.verifyNoInteractions(raceEvents)
    }

    @Test
    fun `untyped admin events still refuse the listing switch`() {
        for (listed in listOf(true, false)) {
            val event = live().apply { createdBy = null }

            assertFailsWith<ValidationException> { service.setListing(adminId, event.id, listed) }

            assertNull(event.hiddenAt)
            assertNull(event.adminApprovedAt)
        }
        Mockito.verifyNoInteractions(decisions)
    }

    private fun submitted(): Event = event(EventStatus.DRAFT, EventReviewStatus.PENDING)
    private fun live(): Event = event(EventStatus.ACTIVE, EventReviewStatus.APPROVED)

    private fun event(status: EventStatus, review: EventReviewStatus): Event {
        val event = Event(
            slug = "fun-run-${UUID.randomUUID().toString().take(6)}",
            name = "Fun Run",
            date = LocalDate.of(2026, 10, 10),
            location = "Cebu City",
            status = status,
            pricePerPhoto = BigDecimal("150.00"),
            createdBy = photographerId,
            reviewStatus = review,
        )
        Mockito.`when`(eventRepository.findByIdForReview(event.id)).thenReturn(event)
        Mockito.`when`(eventRepository.findReviewStatus(event.id)).thenAnswer { event.reviewStatus }
        Mockito.`when`(eventRepository.findById(event.id)).thenReturn(Optional.of(event))
        return event
    }

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
