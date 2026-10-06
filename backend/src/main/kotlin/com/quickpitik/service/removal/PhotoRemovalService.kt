package com.quickpitik.service.removal

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.context.ApplicationEventPublisher
import com.quickpitik.websocket.AdminInboxEvent
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.security.MessageDigest
import java.time.Clock
import java.util.HexFormat
import java.util.UUID

@Service
class PhotoRemovalService(
    private val requests: PhotoRemovalRequestRepository,
    private val commands: PhotoRemovalCommandRepository,
    private val events: EventRepository,
    private val users: UserRepository,
    private val items: PhotoRemovalItemRepository,
    private val photos: PhotoRepository,
    private val reviews: PhotoRemovalReviewRepository,
    private val notifier: PhotoRemovalNotifier,
    private val availability: PhotoAvailabilityService,
    private val evidence: PhotoRemovalEvidenceService,
    private val queries: PhotoRemovalQueryService,
    private val history: PhotoRemovalHistoryRecorder,
    private val transitions: PhotoRemovalTransitions,
    private val tx: TransactionTemplate,
    private val mapper: ObjectMapper,
    private val clock: Clock,
    private val publisher: ApplicationEventPublisher,
) {
    @Transactional(propagation = Propagation.NEVER)
    fun respond(photographerId: UUID, requestId: UUID, command: RespondToRemovalCommand): PhotographerRemovalRequestDto {
        val hash = commandHash(mapper, command)
        return tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            val owned = items.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(requestId, photographerId).associateBy { it.id }
            if (owned.isEmpty()) removalNotFound()
            fun PhotoRemovalCommand.replayOwner(): PhotographerRemovalRequestDto {
                ensureIdentity(photographerId, requestId, RemovalCommandOperation.RESPOND, hash)
                if (state != RemovalCommandState.SUCCEEDED || result == null) removalConflict()
                val replay = mapper.treeToValue(result, PhotographerRemovalRequestDto::class.java)
                return replay.copy(items = replay.items.map { it.copy(photo = it.photo?.let { photo -> photo.copy(price = photo.price.setScale(2)) }) })
            }
            commands.findByCommandId(command.commandId)?.let { return@execute it.replayOwner() }
            transitions.ensureVersion(request, command.expectedVersion)
            if (request.status != RemovalRequestStatus.OPEN || commands.findByRequestIdAndStateIn(requestId,
                    listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN)).isNotEmpty()) removalConflict()
            if (command.itemIds.isEmpty() || command.itemIds.size > 100 || command.itemIds.toSet().size != command.itemIds.size)
                removalInvalid("Select one to 100 distinct items")
            val selected = command.itemIds.map { owned[it] ?: removalNotFound() }
            if (selected.any { !it.awaitingDecision() }) removalConflict()
            if (command.response == PhotographerRemovalResponse.OBJECT && command.reason.isNullOrBlank()) removalInvalid("An objection requires a reason")
            command.reason?.let { validateText(it, 2000) }
            val now = clock.instant()
            if (commands.insertClaim(command.commandId, requestId, RemovalCommandOperation.RESPOND.name, photographerId, hash, null, now) == 0)
                return@execute commands.findByCommandId(command.commandId)!!.replayOwner()
            val observed = if (command.response == PhotographerRemovalResponse.ALREADY_DELETED)
                photos.findAllByIdForUpdate(selected.mapNotNull { it.photoId }.sorted()).associateBy { it.id } else emptyMap()
            selected.forEach { item ->
                val previous = item.ownerResponse
                item.ownerResponse = command.response
                item.ownerResponseReason = command.reason
                item.ownerRespondedAt = now
                val photo = observed[item.photoId]
                val metadata = if (command.response == PhotographerRemovalResponse.ALREADY_DELETED)
                    mapOf("observedPhotoState" to observedRemovalPhotoState(photo)) else emptyMap()
                history.record(RemovalHistoryEvent(requestId, item.id, item.reviewId, item.photoIdSnapshot,
                    RemovalActorType.PHOTOGRAPHER, photographerId, RemovalHistoryActions.OWNER_RESPONDED,
                    previous?.name, command.response.name, command.reason, now, command.commandId, metadata))
            }
            items.flush()
            requests.advanceVersion(requestId, now)
            val result = queries.photographerSnapshot(photographerId, requestId).dto
            val receipt = commands.findByCommandId(command.commandId)!!
            receipt.result = mapper.valueToTree(result)
            receipt.state = RemovalCommandState.SUCCEEDED
            commands.flush()
            publisher.publishEvent(removalChangedEvent(requestId, photographerId, now))
            result
        }!!
    }

    @Transactional(propagation = Propagation.NEVER)
    fun verify(adminId: UUID, requestId: UUID, command: VerifyRemovalCommand): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, command)
        val result = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(command.commandId)?.let {
                return@execute it.replay(adminId, requestId, RemovalCommandOperation.VERIFY, hash, mapper)
            }
            transitions.ensureVersion(request, command.expectedVersion)
            if (request.status != RemovalRequestStatus.DRAFT || request.verifiedAt != null || request.eventId == null ||
                commands.findByRequestIdAndStateIn(requestId, listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN)).isNotEmpty()) removalConflict()
            validateText(command.verificationMethod, 2000)
            validateText(command.verificationNote, 2000)
            if (command.selectedPhotos.isEmpty() || command.selectedPhotos.size > 100 ||
                command.selectedPhotos.map { it.photoId }.toSet().size != command.selectedPhotos.size) removalInvalid("Select one to 100 distinct photos")
            command.selectedPhotos.forEach {
                if (it.source == RemovalMatchSource.MANUAL && it.note.isNullOrBlank()) removalInvalid("Manual selections require an admin note")
                it.note?.let { note -> validateText(note, 2000) }
            }
            val selectedIds = command.selectedPhotos.map { it.photoId }.sorted()
            val fresh = photos.findAllByIdForUpdate(selectedIds).associateBy { it.id }
            if (fresh.size != selectedIds.size || fresh.values.any { it.eventId != request.eventIdSnapshot }) removalInvalid("Selected photos must belong to this event")
            val cache = request.candidateCache
            val scores = command.selectedPhotos.associate { selection ->
                val candidate = if (selection.source == RemovalMatchSource.SELFIE) {
                    if (request.evidenceKey == null || cache == null || cache.requestVersion != request.version ||
                        cache.evidenceGeneration != request.evidenceGeneration) removalConflict()
                    cache.candidates.singleOrNull { it.photoId == selection.photoId && it.source == RemovalMatchSource.SELFIE }
                        ?: removalInvalid("Selfie selection must use this case's latest candidates")
                } else null
                selection.photoId to candidate?.score
            }
            val now = clock.instant()
            if (commands.insertClaim(command.commandId, requestId, RemovalCommandOperation.VERIFY.name, adminId, hash, null, now) == 0) {
                return@execute commands.findByCommandId(command.commandId)!!.replay(adminId, requestId, RemovalCommandOperation.VERIFY, hash, mapper)
            }
            val owners = users.findAllById(fresh.values.mapNotNull { it.photographerId }.toSet()).associateBy { it.id }
            val groups = fresh.values.mapNotNull { it.photographerId }.toSet().associateWith { ownerId ->
                reviews.save(PhotoRemovalReview(requestId = requestId, ownerIdSnapshot = ownerId, ownerId = owners[ownerId]?.id))
            }
            reviews.flush()
            val confirmed = command.selectedPhotos.map { selection ->
                val photo = fresh.getValue(selection.photoId)
                items.save(PhotoRemovalItem(requestId = requestId, photoIdSnapshot = photo.id, photoId = photo.id,
                    ownerIdSnapshot = photo.photographerId, ownerNameSnapshot = owners[photo.photographerId]?.name,
                    reviewId = groups[photo.photographerId]?.id, state = RemovalItemState.VERIFIED, matchSource = selection.source,
                    matchScore = scores[photo.id], verificationActorId = adminId, verifiedAt = now,
                    verificationReason = selection.note ?: command.verificationNote))
            }
            items.flush()
            request.status = RemovalRequestStatus.OPEN
            request.verifierId = adminId
            request.verificationMethod = command.verificationMethod
            request.verificationNote = command.verificationNote
            request.verifiedAt = now
            request.updatedAt = now
            requests.flush()
            confirmed.forEach { item ->
                history.record(RemovalHistoryEvent(requestId = requestId, itemId = item.id, reviewId = item.reviewId,
                    photoIdSnapshot = item.photoIdSnapshot, actorType = RemovalActorType.ADMIN, actorId = adminId,
                    action = RemovalHistoryActions.VERIFIED, toState = RemovalItemState.VERIFIED.name, reason = item.verificationReason,
                    occurredAt = now, commandId = command.commandId))
                if (item.ownerIdSnapshot == null) history.record(RemovalHistoryEvent(requestId = requestId, itemId = item.id,
                    photoIdSnapshot = item.photoIdSnapshot, actorType = RemovalActorType.SYSTEM,
                    action = RemovalHistoryActions.OWNER_UNAVAILABLE, reason = "Photo has no assigned owner; admin action required",
                    occurredAt = now, commandId = UUID.nameUUIDFromBytes("${command.commandId}:orphan:${item.id}".toByteArray(Charsets.UTF_8))))
            }
            // Conditional availability writer clears the persistence context after its update.
            availability.refreshRemovalState(selectedIds.toSet())
            val detail = queries.adminDetail(adminId, requestId)
            val receipt = commands.findByCommandId(command.commandId)!!
            receipt.result = mapper.valueToTree(detail)
            receipt.state = RemovalCommandState.SUCCEEDED
            commands.flush()
            detail
        }!!
        // Inbox persistence gets its own atomic message/deadline/history transaction. Failed owners stay due.
        result.reviews.filter { it.notifiedAt == null && it.unavailableReason == null }.forEach { notifier.attempt(it.id) }
        return result
    }

    fun create(adminId: UUID, command: CreateRemovalRequestCommand): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, command)
        fun attempt(): PhotoRemovalRequestDto = tx.execute {
            commands.findByCommandId(command.commandId)?.let { return@execute it.replay(adminId, null, RemovalCommandOperation.CREATE, hash, mapper) }
            validateText(command.supportReference, 1000)
            validateText(command.requesterContact, 1000)
            validateText(command.reason, 2000)
            command.adminNote?.let { validateText(it, 2000) }
            val event = events.findById(command.eventId).orElse(null) ?: removalNotFound()
            if (event.deletedAt != null) removalNotFound()
            command.requesterUserId?.let {
                if (users.findById(it).orElse(null)?.role != Role.RUNNER) removalInvalid("Requester account must be a runner")
            }
            val now = clock.instant()
            val request = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id, eventIdSnapshot = event.id,
                eventNameSnapshot = event.name, requesterUserId = command.requesterUserId, requesterContact = command.requesterContact,
                supportReference = command.supportReference, requesterReason = command.reason, publicToOwnerReason = "Privacy removal request",
                creatorId = adminId, creationCommandId = command.commandId, createdAt = now, verificationNote = command.adminNote))
            if (commands.insertClaim(command.commandId, request.id, RemovalCommandOperation.CREATE.name, adminId, hash, null, now) == 0) {
                commands.findByCommandId(command.commandId)!!.replay(adminId, null, RemovalCommandOperation.CREATE, hash, mapper)
                // An identical CREATE already owns the creation-command unique key above.
                removalConflict()
            }
            history.record(RemovalHistoryEvent(requestId = request.id, actorType = RemovalActorType.ADMIN, actorId = adminId,
                action = RemovalHistoryActions.CREATED, toState = request.status.name, reason = command.adminNote,
                occurredAt = now, commandId = command.commandId))
            val result = queries.adminDetail(adminId, request.id)
            val receipt = commands.findByCommandId(command.commandId)!!
            receipt.result = mapper.valueToTree(result)
            receipt.state = RemovalCommandState.SUCCEEDED
            commands.flush()
            result
        }!!
        return try { attempt() } catch (error: DataIntegrityViolationException) {
            // A concurrent identical create can win the creation-command unique constraint.
            tx.execute { commands.findByCommandId(command.commandId)?.replay(adminId, null, RemovalCommandOperation.CREATE, hash, mapper) ?: throw error }!!
        }
    }

    fun cancel(adminId: UUID, requestId: UUID, command: CancelRemovalCommand): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, command)
        val replay = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(command.commandId)?.let { return@execute it.replay(adminId, requestId, RemovalCommandOperation.CANCEL, hash, mapper) }
            transitions.ensureVersion(request, command.expectedVersion)
            validateText(command.reason, 2000)
            if (request.status !in setOf(RemovalRequestStatus.DRAFT, RemovalRequestStatus.OPEN)) removalConflict()
            val selected = items.findByRequestIdOrderByIdAsc(requestId)
            if (selected.any { it.state in setOf(RemovalItemState.REMOVAL_AUTHORIZED, RemovalItemState.REMOVED) }) removalConflict()
            val now = clock.instant()
            if (commands.insertClaim(command.commandId, requestId, RemovalCommandOperation.CANCEL.name, adminId, hash, null, now) == 0) {
                return@execute commands.findByCommandId(command.commandId)!!.replay(adminId, requestId, RemovalCommandOperation.CANCEL, hash, mapper)
            }
            val previous = request.status
            selected.filter { it.state in setOf(RemovalItemState.CANDIDATE, RemovalItemState.VERIFIED, RemovalItemState.RETAINED_RESTRICTED) }
                .forEach { it.state = RemovalItemState.EXCLUDED }
            items.flush()
            request.status = RemovalRequestStatus.RESOLVING
            request.candidateCache = null
            request.updatedAt = now
            request.evidenceCleanupState = RemovalEvidenceCleanupState.PENDING
            request.evidenceCleanupNextAttemptAt = now
            requests.flush()
            // Availability's native update clears the persistence context; persist the case first.
            availability.refreshRemovalState(selected.mapNotNull { it.photoId }.toSet())
            history.record(RemovalHistoryEvent(requestId = requestId, actorType = RemovalActorType.ADMIN, actorId = adminId,
                action = RemovalHistoryActions.CANCEL_REQUESTED, fromState = previous.name, toState = request.status.name,
                reason = command.reason, occurredAt = now, commandId = command.commandId))
            val result = queries.adminDetail(adminId, requestId)
            val receipt = commands.findByCommandId(command.commandId)!!
            receipt.result = mapper.valueToTree(result)
            receipt.state = RemovalCommandState.SUCCEEDED
            commands.flush()
            null
        }
        if (replay != null) return replay
        evidence.purgeResolvedEvidence(requestId)
        // Receipt is the original command result, before asynchronous cleanup changes version/state.
        return tx.execute { commands.findByCommandId(command.commandId)!!.replay(adminId, requestId, RemovalCommandOperation.CANCEL, hash, mapper) }!!
    }
}

internal fun validateText(value: String, max: Int) {
    if (value.isBlank() || value.length > max) removalInvalid("Photo removal field is empty or too long")
}
internal fun removalChangedEvent(requestId: UUID, actorId: UUID?, now: java.time.Instant) = AdminInboxEvent(mapOf(
    "type" to "photo_removal_changed", "entityId" to requestId.toString(), "actorId" to actorId?.toString(), "occurredAt" to now.toString()))
internal fun commandHash(mapper: ObjectMapper, input: Any): String = hashBytes(mapper.writeValueAsBytes(input))
internal fun hashBytes(bytes: ByteArray): String = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes))
internal fun PhotoRemovalCommand.replay(actor: UUID, request: UUID?, operation: RemovalCommandOperation, hash: String, mapper: ObjectMapper): PhotoRemovalRequestDto {
    ensureIdentity(actor, request, operation, hash)
    if (state != RemovalCommandState.SUCCEEDED || result == null) removalConflict()
    return mapper.treeToValue(result, PhotoRemovalRequestDto::class.java)
}
internal fun PhotoRemovalCommand.ensureIdentity(actor: UUID, request: UUID?, operation: RemovalCommandOperation, hash: String) {
    if (actorIdSnapshot != actor || (request != null && requestId != request) || this.operation != operation || inputHash != hash) removalConflict()
}
