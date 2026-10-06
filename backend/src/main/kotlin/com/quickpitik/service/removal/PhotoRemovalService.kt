package com.quickpitik.service.removal

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
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
    private val availability: PhotoAvailabilityService,
    private val evidence: PhotoRemovalEvidenceService,
    private val queries: PhotoRemovalQueryService,
    private val history: PhotoRemovalHistoryRecorder,
    private val transitions: PhotoRemovalTransitions,
    private val tx: TransactionTemplate,
    private val mapper: ObjectMapper,
    private val clock: Clock,
) {
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
                action = RemovalHistoryActions.CREATED, toState = request.status.name, occurredAt = now, commandId = command.commandId))
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
