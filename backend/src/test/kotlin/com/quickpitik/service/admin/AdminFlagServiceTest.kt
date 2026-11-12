package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.CopySampleDto
import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.BatchMetadataRow
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CopySampleRow
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.HiddenPhotoCount
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import java.math.BigDecimal
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// The flag queue mutates photos (hide → HIDDEN, dismiss/resolve of a hidden
// flag → LIVE again). These pin the transition table and the cascade so a
// mis-click can't silently strand a photo, and hydrate's fallbacks so the
// admin card never renders a null handle.
class AdminFlagServiceTest {

    private lateinit var flagRepository: FlagRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var userRepository: UserRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var storageService: StorageService
    private lateinit var adminDecisionLogRepository: AdminDecisionLogRepository
    private lateinit var photographerMessageRepository: PhotographerMessageRepository
    private lateinit var eventPublisher: ApplicationEventPublisher
    private lateinit var batchRelevanceRepository: BatchRelevanceRepository

    private val adminId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        flagRepository = Mockito.mock(FlagRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        adminDecisionLogRepository = Mockito.mock(AdminDecisionLogRepository::class.java)
        photographerMessageRepository = Mockito.mock(PhotographerMessageRepository::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
        batchRelevanceRepository = Mockito.mock(BatchRelevanceRepository::class.java)
        Mockito.`when`(adminDecisionLogRepository.save(anyArg())).thenAnswer { it.getArgument(0) }
        Mockito.`when`(photographerMessageRepository.save(anyArg())).thenAnswer { it.getArgument(0) }
        Mockito.`when`(eventRepository.findAllById(anyArg())).thenReturn(emptyList())
        Mockito.`when`(userRepository.findAllById(anyArg())).thenReturn(emptyList())
        Mockito.`when`(photographerSettingsRepository.findAllById(anyArg())).thenReturn(emptyList())
        Mockito.`when`(photoRepository.findAllById(anyArg())).thenReturn(emptyList())
    }

    // The real decision-log service over mock repositories, so the decision row and the note can be read back.
    private fun service(flagsEnabled: Boolean = true) = AdminFlagService(
        flagRepository,
        photoRepository,
        eventRepository,
        userRepository,
        photographerSettingsRepository,
        storageService,
        StorageProperties(),
        AdminProperties(flagsEnabled = flagsEnabled),
        AdminDecisionLogService(adminDecisionLogRepository, photographerMessageRepository, eventPublisher),
        eventPublisher,
        batchRelevanceRepository,
        RelevanceProperties(),
    )

    private fun photo(status: PhotoStatus = PhotoStatus.LIVE, watermarkKey: String? = null) = Photo(
        eventId = UUID.randomUUID(),
        s3Key = "events/e/photos/p/original.jpg",
        pricePhp = BigDecimal.TEN,
        status = status,
        watermarkS3Key = watermarkKey,
    )

    private fun noOtherHiddenFlag(f: Flag, other: Boolean = false) {
        Mockito.`when`(
            flagRepository.existsByTargetKindWireAndTargetIdAndStatusWireAndIdNot(
                f.targetKindWire, f.targetId, FlagStatus.HIDDEN.wire, f.id,
            ),
        ).thenReturn(other)
    }

    private fun flag(status: FlagStatus, photo: Photo? = null): Flag {
        val f = Flag(
            targetKindWire = FlagTargetKind.PHOTO.wire,
            targetId = photo?.id ?: UUID.randomUUID(),
            reason = "inappropriate",
            statusWire = status.wire,
        )
        Mockito.`when`(flagRepository.findPhotoId(f.id)).thenReturn(f.targetId)
        Mockito.`when`(flagRepository.findByIdForUpdate(f.id)).thenReturn(f)
        if (photo != null) {
            Mockito.`when`(photoRepository.findById(photo.id)).thenReturn(Optional.of(photo))
            Mockito.`when`(photoRepository.findAllById(anyArg())).thenReturn(listOf(photo))
        }
        return f
    }

    @Test
    fun `escalate moves an open flag without stamping a resolution`() {
        val f = flag(FlagStatus.OPEN)
        val dto = service().escalate(adminId, f.id, "needs a second look")
        assertEquals("escalated", dto.status)
        assertNull(f.resolvedAt)
        assertNull(f.resolvedBy)
        assertEquals("needs a second look", dto.reviewerNote)
    }

    @Test
    fun `hide cascades HIDDEN onto the photo through a conditional update`() {
        val p = photo()
        val f = flag(FlagStatus.OPEN, p)
        val dto = service().hide(adminId, f.id, null)
        assertEquals("hidden", dto.status)
        Mockito.verify(photoRepository).hideForFlag(p.id)
        // A full-row save would clobber whatever the watermark pipeline wrote
        // to the same row between the read and the write.
        Mockito.verify(photoRepository, Mockito.never()).save(anyArg<Photo>())
    }

    @Test
    fun `hide of a closed flag is a 409`() {
        val f = flag(FlagStatus.DISMISSED)
        val ex = assertFailsWith<ApiException> { service().hide(adminId, f.id, null) }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        Mockito.verify(flagRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `restoring a watermarked photo returns it to LIVE`() {
        val p = photo(PhotoStatus.HIDDEN, watermarkKey = "events/e/photos/p/wm.jpg")
        val f = flag(FlagStatus.HIDDEN, p)
        noOtherHiddenFlag(f)
        val dto = service().dismiss(adminId, f.id, "false alarm")
        assertEquals("dismissed", dto.status)
        Mockito.verify(photoRepository).unhideForFlag(p.id, PhotoStatus.LIVE)
    }

    @Test
    fun `restoring a photo without a watermark key returns it to PROCESSING`() {
        val p = photo(PhotoStatus.HIDDEN)
        val f = flag(FlagStatus.HIDDEN, p)
        noOtherHiddenFlag(f)
        service().resolve(adminId, f.id, null)
        // LIVE would publish the clean original: with no derived asset,
        // PhotoService.resolveThumbnailUrl falls back to s3Key. PROCESSING
        // hands the row back to the watermark sweep instead.
        Mockito.verify(photoRepository).unhideForFlag(p.id, PhotoStatus.PROCESSING)
        Mockito.verify(photoRepository, Mockito.never()).unhideForFlag(p.id, PhotoStatus.LIVE)
    }

    @Test
    fun `dismissing a hidden flag keeps the photo hidden while another hidden flag remains`() {
        val p = photo(PhotoStatus.HIDDEN)
        val f = flag(FlagStatus.HIDDEN, p)
        noOtherHiddenFlag(f, other = true)
        service().dismiss(adminId, f.id, null)
        Mockito.verify(photoRepository, Mockito.never()).unhideForFlag(anyArg(), anyArg())
    }

    @Test
    fun `list rejects an unknown status before touching the repository`() {
        assertFailsWith<ValidationException> {
            service().list("bogus", null, PaginationParams.of(0, 10))
        }
        Mockito.verifyNoInteractions(flagRepository)
    }

    @Test
    fun `hydrate falls back when the photo has no derived asset and no owner`() {
        val p = photo(PhotoStatus.PROCESSING)
        val f = flag(FlagStatus.OPEN, p)
        val dto = service().escalate(adminId, f.id, null)
        assertNull(dto.photoSnapshot.thumbnailUrl, "a PROCESSING photo must not presign the original")
        assertEquals("photographer", dto.photographerHandle)
        assertEquals("system", dto.reportedBy)
        assertNull(dto.reviewedBy)
        Mockito.verify(storageService, Mockito.never()).presignedGetUrl(anyString(), anyArg())
    }

    // A system or report flag on an event or a photographer, as SystemFlagService writes it.
    private fun targetFlag(kind: FlagTargetKind, targetId: UUID): Flag {
        val f = Flag(targetKindWire = kind.wire, targetId = targetId, reason = "possible_duplicate", note = "pair: Cebu Marathon")
        Mockito.`when`(flagRepository.findByIdForUpdate(f.id)).thenReturn(f)
        return f
    }

    private fun photographer(handle: String, name: String): Pair<User, PhotographerSettings> {
        val user = User(id = UUID.randomUUID(), email = "$handle@example.com", passwordHash = "x", name = name, role = Role.PHOTOGRAPHER)
        val settings = PhotographerSettings(userId = user.id, handle = handle)
        Mockito.`when`(userRepository.findAllById(anyArg())).thenReturn(listOf(user))
        Mockito.`when`(photographerSettingsRepository.findAllById(anyArg())).thenReturn(listOf(settings))
        return user to settings
    }

    @Test
    fun `an event flag shows the event name and its creator`() {
        val (creator, _) = photographer("juanshots", "Juan Dela Cruz")
        val event = Event(slug = "cebu-marathon-2027", name = "Cebu Marathon 2027", date = LocalDate.of(2027, 1, 10), location = "Cebu City", status = EventStatus.ACTIVE, createdBy = creator.id)
        Mockito.`when`(eventRepository.findAllById(anyArg())).thenReturn(listOf(event))
        val f = targetFlag(FlagTargetKind.EVENT, event.id)

        val dto = service().escalate(adminId, f.id, null)

        assertEquals(event.id, dto.eventId)
        assertEquals("Cebu Marathon 2027", dto.eventName)
        assertEquals("juanshots", dto.photographerHandle)
        assertEquals("Juan Dela Cruz", dto.photographerName)
        assertNull(dto.photoId)
        assertEquals("event", dto.targetKind)
        assertEquals("pair: Cebu Marathon", dto.note)
    }

    @Test
    fun `a user flag shows the photographer handle`() {
        val (user, _) = photographer("juanshots", "Juan Dela Cruz")
        val f = targetFlag(FlagTargetKind.USER, user.id)

        val dto = service().escalate(adminId, f.id, null)

        assertEquals("juanshots", dto.photographerHandle)
        assertEquals("Juan Dela Cruz", dto.photographerName)
        assertNull(dto.eventId)
        assertNull(dto.photoId)
        assertEquals("user", dto.targetKind)
    }

    // ─── PE-2.2: the payout hold on a relevance flag ──────────────────────
    private val photographerId = UUID.randomUUID()
    private val relevance = SystemFlagReason.BATCH_RELEVANCE_LOW.wire

    private fun relevanceFlag(status: FlagStatus = FlagStatus.OPEN, held: Boolean = false): Flag {
        val f = Flag(
            targetKindWire = FlagTargetKind.USER.wire,
            targetId = photographerId,
            reason = relevance,
            statusWire = status.wire,
            payoutHoldAt = if (held) OffsetDateTime.now().minusDays(1) else null,
        )
        Mockito.`when`(flagRepository.findRelevanceUserTarget(f.id, relevance)).thenReturn(photographerId)
        Mockito.`when`(flagRepository.findByIdForUpdate(f.id)).thenReturn(f)
        return f
    }

    private fun savedMessages(): List<PhotographerMessage> {
        val captor = ArgumentCaptor.forClass(PhotographerMessage::class.java)
        Mockito.verify(photographerMessageRepository, Mockito.atLeast(0)).save(captor.capture())
        return captor.allValues
    }

    @Test
    fun `confirming a hold stamps who and when, logs a decision and sends one note`() {
        val f = relevanceFlag()

        val dto = service().confirmPayoutHold(adminId, f.id)

        assertNotNull(f.payoutHoldAt)
        assertEquals(adminId, f.payoutHoldBy)
        assertEquals(f.payoutHoldAt.toString(), dto.payoutHoldAt)
        assertEquals("open", dto.status) // the hold leaves the flag open: closing it ends the hold
        val decision = ArgumentCaptor.forClass(AdminDecisionLog::class.java)
        Mockito.verify(adminDecisionLogRepository).save(decision.capture())
        assertEquals("payout_hold_confirmed", decision.value.decision)
        assertEquals(photographerId, decision.value.targetUserId)
        assertEquals("flag ${f.id}", decision.value.reason)
        val note = savedMessages().single()
        assertEquals(photographerId, note.photographerId)
        assertEquals(PhotographerMessageKind.ADMIN_MESSAGE.wire, note.kindWire)
        assertEquals("Payouts on hold", note.title)
        assertEquals(adminId, note.sourceAdminId)
        assertEquals(decision.value.id, note.sourceDecisionId)
        Mockito.verify(eventPublisher).publishEvent(PayoutHoldPlacedEvent(photographerId))
    }

    @Test
    fun `confirming twice changes nothing`() {
        val f = relevanceFlag(held = true)
        val stamped = f.payoutHoldAt

        service().confirmPayoutHold(adminId, f.id)

        assertEquals(stamped, f.payoutHoldAt)
        assertNull(f.payoutHoldBy)
        Mockito.verify(flagRepository, Mockito.never()).save(anyArg())
        Mockito.verifyNoInteractions(adminDecisionLogRepository, photographerMessageRepository, eventPublisher)
    }

    @Test
    fun `a hold cannot be confirmed on a photo flag, an event flag, another reason or a closed flag`() {
        val photoFlag = flag(FlagStatus.OPEN).apply { reason = relevance }
        val eventFlag = targetFlag(FlagTargetKind.EVENT, UUID.randomUUID()).apply { reason = relevance }
        val otherReason = targetFlag(FlagTargetKind.USER, photographerId)
        val closed = listOf(FlagStatus.RESOLVED, FlagStatus.DISMISSED, FlagStatus.HIDDEN).map { relevanceFlag(it) }

        for (f in listOf(photoFlag, eventFlag, otherReason) + closed) {
            val ex = assertFailsWith<ApiException> { service().confirmPayoutHold(adminId, f.id) }
            assertEquals(HttpStatus.CONFLICT, ex.status)
            assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
            assertNull(f.payoutHoldAt)
        }
        Mockito.verify(flagRepository, Mockito.never()).save(anyArg())
        Mockito.verifyNoInteractions(adminDecisionLogRepository, photographerMessageRepository, eventPublisher)
    }

    @Test
    fun `the confirm is refused while the flags switch is off`() {
        val f = relevanceFlag()
        val ex = assertFailsWith<ApiException> { service(flagsEnabled = false).confirmPayoutHold(adminId, f.id) }
        assertEquals(ErrorCodes.FLAGS_DISABLED, ex.code)
        assertNull(f.payoutHoldAt)
    }

    @Test
    fun `the old hide route refuses a relevance flag`() {
        val f = relevanceFlag(held = true)
        val ex = assertFailsWith<ApiException> { service().hide(adminId, f.id, null) }
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertEquals("open", f.statusWire)
        Mockito.verify(flagRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `every action on a relevance flag locks the photographer before the flag`() {
        val actions: List<(UUID) -> Unit> = listOf(
            { id -> service().confirmPayoutHold(adminId, id) },
            { id -> service().resolve(adminId, id, null) },
            { id -> service().dismiss(adminId, id, null) },
            { id -> service().escalate(adminId, id, null) },
        )
        for (action in actions) {
            Mockito.clearInvocations(userRepository, flagRepository)
            val f = relevanceFlag()
            action(f.id)
            val order = Mockito.inOrder(userRepository, flagRepository)
            order.verify(userRepository).lockById(photographerId)
            order.verify(flagRepository).findByIdForUpdate(f.id)
        }
    }

    @Test
    fun `resolving or dismissing a held flag sends the lift note once`() {
        val resolved = relevanceFlag(held = true)
        service().resolve(adminId, resolved.id, "checked")
        val dismissed = relevanceFlag(held = true)
        service().dismiss(adminId, dismissed.id, null)
        val unheld = relevanceFlag()
        service().resolve(adminId, unheld.id, null) // no hold, no lift note
        assertFailsWith<ApiException> { service().resolve(adminId, resolved.id, null) } // already closed

        val notes = savedMessages()
        assertEquals(2, notes.size)
        for (note in notes) {
            assertEquals("Payout hold lifted", note.title)
            assertEquals("The review of your uploads is finished. You can request payouts again.", note.body)
            assertEquals(adminId, note.sourceAdminId)
            assertEquals(photographerId, note.photographerId)
        }
    }

    @Test
    fun `no note names another photographer`() {
        val (other, _) = photographer("othershots", "Maria Santos")
        val f = relevanceFlag().apply { note = "event e1 2027-01-10: copy 9/10 · matches ${other.id} othershots" }
        service().confirmPayoutHold(adminId, f.id)
        service().resolve(adminId, f.id, null)

        val notes = savedMessages()
        assertEquals(2, notes.size)
        for (note in notes) {
            val text = "${note.title} ${note.body}"
            for (forbidden in listOf("othershots", "Maria", other.id.toString(), "copied", "copy")) {
                assertFalse(forbidden in text, "note must not carry '$forbidden': $text")
            }
        }
    }

    // ─── PE-2.3: the evidence behind a relevance flag ─────────────────────
    private val raceDate = LocalDate.of(2027, 1, 10)
    private val race = Event(slug = "cebu-race", name = "Cebu Race 2027", date = raceDate, location = "Cebu City", status = EventStatus.ACTIVE)
    private val otherId = UUID.randomUUID()

    private fun readable(f: Flag): Flag = f.also { Mockito.`when`(flagRepository.findById(it.id)).thenReturn(Optional.of(it)) }

    private fun linkedBatch(f: Flag, event: Event = race, shared: Boolean = true) = BatchRelevance(
        eventId = event.id, photographerId = photographerId,
        level = "LOW", copyLow = true, dateLow = if (shared) true else null, photosLive = 20, lookups = 20, tightHits = 9, weakHits = 2,
        exactMatches = 1, xmpMatches = 0,
        datedPhotos = if (shared) 12 else null, datedOutside = if (shared) 11 else null, metadataPending = if (shared) 0 else null,
        bibbedPhotos = 5, bibOverlapPhotos = 1, scoredAt = OffsetDateTime.parse("2027-01-15T02:00:00Z"),
        flagId = f.id, flaggedAt = OffsetDateTime.parse("2027-01-15T02:00:00Z"),
    ).also { Mockito.`when`(batchRelevanceRepository.findByFlagId(f.id)).thenReturn(listOf(it)) }

    private fun pic(eventId: UUID = race.id, owner: UUID? = photographerId, key: String? = "wm/${UUID.randomUUID()}.jpg") =
        Photo(eventId = eventId, s3Key = "orig/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal.TEN, watermarkS3Key = key)
            .apply { photographerId = owner }

    private fun sample(photo: UUID, match: UUID? = null, distance: Int? = null, pair: String? = null, exact: UUID? = null) =
        object : CopySampleRow {
            override val photoId = photo
            override val matchPhotoId = match
            override val distance = distance
            override val pair = pair
            override val exactPhotoId = exact
        }

    private fun meta(photo: UUID, takenOn: LocalDate?, state: PhotoMetadataState = PhotoMetadataState.READY) = object : BatchMetadataRow {
        override val photoId = photo
        override val state = state
        override val fields = takenOn?.let { PhotoMetadataFields(takenAtLocal = it.atTime(7, 0), takenAtOffset = "+08:00") }
    }

    private fun known(vararg photos: Photo) {
        Mockito.`when`(photoRepository.findAllById(anyArg())).thenReturn(photos.toList())
        Mockito.`when`(storageService.presignedGetUrl(anyString(), anyArg())).thenAnswer { "signed:" + it.getArgument<String>(0) }
        val other = User(id = otherId, email = "o@example.com", passwordHash = "x", name = "Maria Santos", role = Role.PHOTOGRAPHER)
        Mockito.`when`(userRepository.findAllById(anyArg())).thenReturn(listOf(other))
        Mockito.`when`(photographerSettingsRepository.findAllById(anyArg())).thenReturn(listOf(PhotographerSettings(userId = otherId, handle = "othershots")))
    }

    // The single copy sample of a one-batch relevance flag on the race.
    private fun copySampleOf(row: CopySampleRow, vararg photos: Photo): CopySampleDto {
        val f = readable(relevanceFlag())
        val batch = linkedBatch(f)
        Mockito.`when`(eventRepository.findAllById(anyArg())).thenReturn(listOf(race))
        Mockito.`when`(batchRelevanceRepository.copySamples(batch.id, 12)).thenReturn(listOf(row))
        known(*photos)
        return service().relevanceEvidence(f.id).batches.single().copySamples.single()
    }

    @Test
    fun `relevance evidence lists each linked batch with its counts and samples`() {
        val f = readable(relevanceFlag(held = true))
        val batch = linkedBatch(f)
        Mockito.`when`(eventRepository.findAllById(anyArg())).thenReturn(listOf(race))
        val mine = pic()
        val theirs = pic(owner = otherId)
        val inWindow = pic()
        val outside = pic()
        Mockito.`when`(batchRelevanceRepository.copySamples(batch.id, 12)).thenReturn(listOf(sample(mine.id, theirs.id, 2, "CLEAN")))
        Mockito.`when`(batchRelevanceRepository.metadataRows(race.id, photographerId)).thenReturn(listOf(
            meta(inWindow.id, raceDate.plusDays(1)),
            meta(outside.id, LocalDate.of(2026, 12, 1)),
            meta(pic().id, null, PhotoMetadataState.PENDING),
        ))
        Mockito.`when`(flagRepository.countHiddenPhotosByEvent(f.id)).thenReturn(listOf(object : HiddenPhotoCount {
            override val eventId = race.id
            override val hidden = 3L
        }))
        known(mine, theirs, inWindow, outside)

        val dto = service().relevanceEvidence(f.id)

        assertEquals(f.id, dto.flagId)
        assertEquals(f.payoutHoldAt.toString(), dto.payoutHoldAt)
        val b = dto.batches.single()
        assertEquals(race.id, b.eventId)
        assertEquals("Cebu Race 2027", b.eventName)
        assertEquals("2027-01-10", b.eventDate)
        assertTrue(b.sharedRace)
        assertEquals("LOW", b.level)
        assertEquals(listOf(20, 20, 9, 2, 1, 0), listOf(b.photosLive, b.lookups, b.tightHits, b.weakHits, b.exactMatches, b.xmpMatches))
        assertEquals(listOf(12, 11, 0, 5, 1), listOf(b.datedPhotos, b.datedOutside, b.metadataPending, b.bibbedPhotos, b.bibOverlapPhotos))
        assertEquals("2027-01-15T02:00Z", b.scoredAt)
        assertEquals(3, b.hiddenPhotos)
        val copy = b.copySamples.single()
        assertEquals(mine.id, copy.photoId)
        assertEquals("signed:${mine.watermarkS3Key}", copy.thumbnailUrl)
        assertEquals(theirs.id, copy.matchPhotoId)
        assertEquals("signed:${theirs.watermarkS3Key}", copy.matchThumbnailUrl)
        assertEquals("othershots", copy.matchPhotographerHandle)
        assertEquals(2, copy.distance)
        assertEquals("CLEAN", copy.pair)
        assertFalse(copy.exact)
        val date = b.dateSamples.single() // only the photo outside the window
        assertEquals(outside.id, date.photoId)
        assertEquals("2026-12-01", date.takenOn)
        assertEquals("signed:${outside.watermarkS3Key}", date.thumbnailUrl)
    }

    @Test
    fun `a copy sample names the other photographer and a deleted match shows no thumbnail`() {
        val mine = pic()
        val trashed = pic(owner = otherId).apply { deletedAt = OffsetDateTime.now() }
        val copy = copySampleOf(sample(mine.id, trashed.id, 1, "CENTRE"), mine, trashed)
        assertEquals(trashed.id, copy.matchPhotoId)
        assertNull(copy.matchThumbnailUrl)
        assertEquals("othershots", copy.matchPhotographerHandle)
    }

    @Test
    fun `a match whose row is gone keeps its id and shows no thumbnail and no handle`() {
        val mine = pic()
        val gone = UUID.randomUUID()
        val copy = copySampleOf(sample(mine.id, gone, 0, "CLEAN"), mine)
        assertEquals(gone, copy.matchPhotoId)
        assertNull(copy.matchThumbnailUrl)
        assertNull(copy.matchPhotographerHandle)
    }

    @Test
    fun `a match kept as a privacy tombstone shows no thumbnail`() {
        val mine = pic()
        val tombstone = pic(owner = otherId, key = null).apply { deletedAt = OffsetDateTime.now() }
        val copy = copySampleOf(sample(mine.id, tombstone.id, 3, "MARKED"), mine, tombstone)
        assertNull(copy.matchThumbnailUrl)
    }

    @Test
    fun `a match whose photographer is gone shows no handle`() {
        val mine = pic()
        val ownerless = pic(owner = null)
        val copy = copySampleOf(sample(mine.id, ownerless.id, 0, "CLEAN"), mine, ownerless)
        assertEquals("signed:${ownerless.watermarkS3Key}", copy.matchThumbnailUrl)
        assertNull(copy.matchPhotographerHandle)
    }

    @Test
    fun `an exact copy in another event names its photographer and its event`() {
        val mine = pic()
        val elsewhere = UUID.randomUUID()
        val original = pic(eventId = elsewhere, owner = otherId)
        val copy = copySampleOf(sample(mine.id, exact = original.id), mine, original)
        assertTrue(copy.exact)
        assertNull(copy.matchPhotoId)
        assertNull(copy.distance)
        assertEquals(original.id, copy.exactPhotoId)
        assertEquals("signed:${original.watermarkS3Key}", copy.exactThumbnailUrl)
        assertEquals("othershots", copy.exactPhotographerHandle)
        assertEquals(elsewhere, copy.exactEventId)
    }

    @Test
    fun `an own event carries no date counts`() {
        val own = Event(slug = "own", name = "My Shoot", date = raceDate, location = "Cebu", status = EventStatus.ACTIVE, createdBy = photographerId)
        val f = readable(relevanceFlag())
        linkedBatch(f, own, shared = false)
        Mockito.`when`(eventRepository.findAllById(anyArg())).thenReturn(listOf(own))

        val b = service().relevanceEvidence(f.id).batches.single()

        assertFalse(b.sharedRace)
        assertNull(b.datedPhotos)
        assertNull(b.datedOutside)
        assertNull(b.dateLow)
        assertTrue(b.dateSamples.isEmpty())
        Mockito.verify(batchRelevanceRepository, Mockito.never()).metadataRows(anyArg(), anyArg())
    }

    @Test
    fun `a flag of another kind has no batches`() {
        val photoFlag = readable(flag(FlagStatus.OPEN))
        val otherReason = readable(targetFlag(FlagTargetKind.USER, photographerId))

        for (f in listOf(photoFlag, otherReason)) {
            val dto = service().relevanceEvidence(f.id)
            assertEquals(f.id, dto.flagId)
            assertTrue(dto.batches.isEmpty())
        }
        Mockito.verifyNoInteractions(batchRelevanceRepository)
    }

    @Test
    fun `the evidence read answers not found for an unknown flag and is refused while the flags switch is off`() {
        val missing = UUID.randomUUID()
        Mockito.`when`(flagRepository.findById(missing)).thenReturn(Optional.empty())
        val notFound = assertFailsWith<NotFoundException> { service().relevanceEvidence(missing) }
        assertEquals(ErrorCodes.FLAG_NOT_FOUND, notFound.code)
        val off = assertFailsWith<ApiException> { service(flagsEnabled = false).relevanceEvidence(missing) }
        assertEquals(ErrorCodes.FLAGS_DISABLED, off.code)
    }
}
