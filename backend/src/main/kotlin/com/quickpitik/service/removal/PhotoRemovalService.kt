package com.quickpitik.service.removal

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import com.quickpitik.service.photos.PhotoErasureService
import com.quickpitik.service.orders.OrderPhotoRemovalService
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.dto.photos.*
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
    private val erasure: PhotoErasureService,
    private val jobs: PhotoErasureJobRepository,
    private val orderRemoval: OrderPhotoRemovalService,
    private val adminLog: AdminDecisionLogService,
    private val audit: PhotoRemovalHistoryRepository,
) {
    @Transactional(propagation = Propagation.NEVER)
    fun decide(adminId: UUID, requestId: UUID, command: DecideRemovalCommand): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, command)
        val result = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(command.commandId)?.let {
                return@execute it.replay(adminId, requestId, RemovalCommandOperation.DECIDE, hash, mapper)
            }
            transitions.ensureVersion(request, command.expectedVersion)
            if (request.status !in setOf(RemovalRequestStatus.OPEN, RemovalRequestStatus.RESOLVING, RemovalRequestStatus.RESOLVED) ||
                commands.findByRequestIdAndStateIn(requestId, listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN)).isNotEmpty()) removalConflict()
            if (command.decisions.isEmpty() || command.decisions.size > 100 || command.decisions.map { it.itemId }.toSet().size != command.decisions.size)
                removalInvalid("Select one to 100 distinct items")
            command.urgentReason?.let { validateText(it, 2000) }
            val selected = command.decisions.map { decision ->
                validateText(decision.reason, 2000)
                items.findByIdAndRequestId(decision.itemId, requestId) ?: removalNotFound()
            }
            // While another photo's cleanup retries, only restricted exceptions stay revisitable; never during cancellation.
            if (request.status == RemovalRequestStatus.RESOLVING && (selected.any { it.state != RemovalItemState.RETAINED_RESTRICTED } ||
                    audit.existsByRequestIdAndAction(requestId, RemovalHistoryActions.CANCEL_REQUESTED))) removalConflict()
            val current = photos.findAllByIdForUpdate(selected.map { it.photoIdSnapshot }.sorted()).associateBy { it.id }
            val groups = reviews.findByRequestIdOrderByIdAsc(requestId).associateBy { it.id }
            val now = clock.instant()
            selected.zip(command.decisions).forEach { (item, decision) ->
                if (item.state !in setOf(RemovalItemState.VERIFIED, RemovalItemState.RETAINED_RESTRICTED)) removalConflict()
                val photo = current[item.photoIdSnapshot]
                if (photo != null && (photo.eventId != request.eventIdSnapshot || photo.photographerId != groups[item.reviewId]?.ownerIdSnapshot))
                    removalConflict()
                if (decision.decision == RemovalDecision.RETAIN_RESTRICTED && decision.retentionReviewAt?.isAfter(now) != true)
                    removalInvalid("Restricted retention requires a future review date")
                if (decision.decision != RemovalDecision.RETAIN_RESTRICTED && decision.retentionReviewAt != null)
                    removalInvalid("Only restricted retention has a review date")
                if (decision.decision == RemovalDecision.REMOVE) {
                    if (photo == null && !jobs.findByPhotoId(item.photoIdSnapshot).confirms(item.photoIdSnapshot)) removalConflict()
                    if (item.ownerResponse != PhotographerRemovalResponse.AGREE &&
                        groups[item.reviewId]?.deadline?.let { it > now } != false && command.urgentReason.isNullOrBlank())
                        removalInvalid("A decision before the response deadline or without a reachable owner requires an urgent reason")
                }
            }
            if (commands.insertClaim(command.commandId, requestId, RemovalCommandOperation.DECIDE.name, adminId, hash, null, now) == 0)
                return@execute commands.findByCommandId(command.commandId)!!.replay(adminId, requestId, RemovalCommandOperation.DECIDE, hash, mapper)
            selected.zip(command.decisions).forEach { (item, decision) ->
                val previous = item.state
                item.retentionReviewAt = decision.retentionReviewAt
                transitions.apply(item, decision.decision, decision.reason)
                item.adminDecisionActorId = adminId
                item.adminDecidedAt = now
                items.saveAndFlush(item)
                if (decision.decision == RemovalDecision.REMOVE) {
                    val photo = current[item.photoIdSnapshot]
                    val jobId = if (photo != null) {
                        availability.authorizeRemoval(photo.id, item.id, adminId)
                        val job = erasure.request(PhotoErasureCommand(photo.id, PhotoErasurePurpose.PRIVACY_REMOVAL,
                            RemovalActorType.ADMIN, adminId, decision.reason, requestId, item.id, command.commandId))
                        orderRemoval.onRemovalAuthorized(photo.id, requestId, now)
                        job
                    } else jobs.findByPhotoId(item.photoIdSnapshot)!!.id
                    items.findById(item.id).orElseThrow().cleanupJobId = jobId
                    items.flush()
                }
                val reason = decision.reason + (command.urgentReason?.let { "\nUrgent decision: $it" } ?: "")
                val logId = adminLog.logPhotoRemovalDecision(adminId, requestId, item.id, decision.decision, reason, command.commandId)
                history.record(RemovalHistoryEvent(requestId, item.id, item.reviewId, item.photoIdSnapshot,
                    RemovalActorType.ADMIN, adminId, RemovalHistoryActions.ADMIN_DECIDED, previous.name, item.state.name,
                    reason, now, command.commandId, mapOf("decisionVersion" to (command.expectedVersion + 1).toString(),
                        "decision" to decision.decision.name, "decisionLogId" to logId.toString())))
            }
            availability.refreshRemovalState(selected.map { it.photoIdSnapshot }.toSet())
            requests.advanceVersion(requestId, now)
            val detail = queries.adminDetail(adminId, requestId)
            val receipt = commands.findByCommandId(command.commandId)!!
            receipt.result = mapper.valueToTree(detail)
            receipt.state = RemovalCommandState.SUCCEEDED
            commands.flush()
            publisher.publishEvent(removalChangedEvent(requestId, adminId, now))
            detail
        }!!
        val selectedIds = command.decisions.map { it.itemId }.toSet()
        result.items.filter { it.id in selectedIds }.mapNotNull { it.cleanupJobId }.distinct().forEach { job ->
            runCatching { erasure.attempt(job) }.onFailure { log.warn("Removal cleanup remains pending for job {} ({})", job, it.javaClass.simpleName) }
        }
        reconcileAffected(requestId, result.items.filter { it.id in selectedIds }.map { it.photoIdSnapshot })
        return result
    }

    @Transactional(propagation = Propagation.NEVER)
    fun retryCleanup(adminId: UUID, requestId: UUID, command: RetryRemovalCleanupCommand): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, command)
        val result = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(command.commandId)?.let {
                return@execute it.replay(adminId, requestId, RemovalCommandOperation.RETRY_CLEANUP, hash, mapper)
            }
            transitions.ensureVersion(request, command.expectedVersion)
            if (command.itemIds.isEmpty() || command.itemIds.size > 100 || command.itemIds.toSet().size != command.itemIds.size)
                removalInvalid("Select one to 100 distinct items")
            val selected = command.itemIds.map { items.findByIdAndRequestId(it, requestId) ?: removalNotFound() }
            photos.findAllByIdForUpdate(selected.map { it.photoIdSnapshot }.sorted())
            selected.forEach { item ->
                val job = item.cleanupJobId?.let { jobs.lock(it) } ?: removalConflict()
                if (job.photoId != item.photoIdSnapshot || job.outcome == PhotoErasureOutcome.COMPLETE ||
                    item.state !in setOf(RemovalItemState.REMOVAL_AUTHORIZED, RemovalItemState.REMOVED)) removalConflict()
            }
            val now = clock.instant()
            if (commands.insertClaim(command.commandId, requestId, RemovalCommandOperation.RETRY_CLEANUP.name, adminId, hash, null, now) == 0)
                return@execute commands.findByCommandId(command.commandId)!!.replay(adminId, requestId, RemovalCommandOperation.RETRY_CLEANUP, hash, mapper)
            selected.forEach { item -> history.record(RemovalHistoryEvent(requestId, item.id, item.reviewId, item.photoIdSnapshot,
                RemovalActorType.ADMIN, adminId, RemovalHistoryActions.ERASURE_PROGRESS, occurredAt = now, commandId = command.commandId,
                metadata = mapOf("kind" to "retry_requested", "jobId" to item.cleanupJobId.toString()))) }
            requests.advanceVersion(requestId, now)
            val detail = queries.adminDetail(adminId, requestId)
            commands.findByCommandId(command.commandId)!!.apply { result = mapper.valueToTree(detail); state = RemovalCommandState.SUCCEEDED }
            commands.flush()
            detail
        }!!
        result.items.filter { it.id in command.itemIds }.mapNotNull { it.cleanupJobId }.distinct().forEach { job ->
            runCatching { erasure.attempt(job) }.onFailure { log.warn("Removal retry remains pending for job {} ({})", job, it.javaClass.simpleName) }
        }
        reconcileAffected(requestId, result.items.filter { it.id in command.itemIds }.map { it.photoIdSnapshot })
        return result
    }

    @Transactional(propagation = Propagation.NEVER)
    fun supportUpdate(adminId: UUID, requestId: UUID, command: RemovalSupportUpdateCommand): RemovalHistoryDto {
        val hash = commandHash(mapper, command)
        return tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            fun PhotoRemovalCommand.replaySupport(): RemovalHistoryDto {
                ensureIdentity(adminId, requestId, RemovalCommandOperation.SUPPORT_UPDATE, hash)
                if (state != RemovalCommandState.SUCCEEDED || result == null) removalConflict()
                return mapper.treeToValue(result, RemovalHistoryDto::class.java)
            }
            commands.findByCommandId(command.commandId)?.let { return@execute it.replaySupport() }
            transitions.ensureVersion(request, command.expectedVersion)
            validateText(command.note, 2000)
            val notice = command.noticeId?.let { id -> audit.findGuestBuyerNotices(requestId).singleOrNull { it.noticeId == id } ?: removalNotFound() }
            val now = clock.instant()
            if (commands.insertClaim(command.commandId, requestId, RemovalCommandOperation.SUPPORT_UPDATE.name, adminId, hash, null, now) == 0)
                return@execute commands.findByCommandId(command.commandId)!!.replaySupport()
            history.record(RemovalHistoryEvent(requestId, actorType = RemovalActorType.SUPPORT, actorId = adminId,
                action = RemovalHistoryActions.SUPPORT_UPDATED, reason = command.note, occurredAt = now, commandId = command.commandId,
                metadata = notice?.let { mapOf("noticeId" to it.noticeId.toString(), "orderId" to it.orderId.toString(), "photoId" to it.photoId.toString()) } ?: emptyMap()))
            requests.advanceVersion(requestId, now)
            val saved = audit.findByRequestIdAndCommandId(requestId, command.commandId).single()
            val result = RemovalHistoryDto(saved.id, requestId, saved.itemId, saved.reviewId, saved.photoIdSnapshot,
                saved.actorType, saved.actorIdSnapshot, saved.action, saved.fromState, saved.toState, saved.reason, saved.occurredAt, saved.commandId, saved.metadata)
            commands.findByCommandId(command.commandId)!!.apply { this.result = mapper.valueToTree(result); state = RemovalCommandState.SUCCEEDED }
            commands.flush()
            publisher.publishEvent(removalChangedEvent(requestId, adminId, now))
            result
        }!!
    }

    private fun reconcileAffected(requestId: UUID, photoIds: List<UUID>) {
        // Never take another case lock while holding the authorizing case/photo locks.
        val affected = if (photoIds.isEmpty()) emptyList() else items.findOverlappingRequestIds(photoIds, org.springframework.data.domain.PageRequest.of(0, 100))
        (listOf(requestId) + affected).distinct().forEach { id ->
            runCatching { reconcile(id) }.onFailure { log.warn("Removal reconciliation remains pending for case {} ({})", id, it.javaClass.simpleName) }
        }
    }

    @Transactional(propagation = Propagation.NEVER)
    fun reconcile(requestId: UUID): PhotoRemovalRequestDto {
        val purge = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            if (request.status == RemovalRequestStatus.CANCELLED || request.status == RemovalRequestStatus.DRAFT) return@execute false
            val selected = items.findByRequestIdOrderByIdAsc(requestId)
            photos.findAllByIdForUpdate(selected.map { it.photoIdSnapshot }.sorted())
            val now = clock.instant()
            var changed = false
            selected.filter { it.state != RemovalItemState.EXCLUDED && it.state != RemovalItemState.CANDIDATE }.forEach { item ->
                val job = jobs.findByPhotoId(item.photoIdSnapshot)?.let { jobs.lock(it.id) }
                if (job.confirms(item.photoIdSnapshot) && (item.state != RemovalItemState.REMOVED || item.cleanupJobId != job!!.id)) {
                    val previous = item.state
                    item.state = RemovalItemState.REMOVED
                    item.cleanupJobId = job!!.id
                    items.flush()
                    history.record(RemovalHistoryEvent(requestId, item.id, item.reviewId, item.photoIdSnapshot,
                        RemovalActorType.SYSTEM, action = RemovalHistoryActions.ERASURE_PROGRESS,
                        fromState = previous.name, toState = item.state.name, occurredAt = now,
                        commandId = UUID.nameUUIDFromBytes("removal:proof:${item.id}:${job.id}:${job.revision}:${job.completedAt}".toByteArray()),
                        metadata = mapOf("jobId" to job.id.toString(), "authorizingRequestId" to (job.requestId?.toString() ?: "ordinary_deletion"),
                            "decisionVersion" to (request.version + 1).toString())))
                    changed = true
                } else if (item.state == RemovalItemState.REMOVED && !job.confirms(item.photoIdSnapshot)) {
                    item.state = RemovalItemState.REMOVAL_AUTHORIZED
                    request.status = RemovalRequestStatus.RESOLVING
                    request.resolvedAt = null
                    changed = true
                }
            }
            if (request.status == RemovalRequestStatus.RESOLVED && selected.any { it.state == RemovalItemState.REMOVAL_AUTHORIZED }) {
                request.status = RemovalRequestStatus.RESOLVING
                request.resolvedAt = null
                changed = true
            }
            val decisioned = selected.none { it.state in setOf(RemovalItemState.CANDIDATE, RemovalItemState.VERIFIED) } &&
                items.findPendingOwnerReviewItemIds(requestId).isEmpty()
            if (decisioned && request.status == RemovalRequestStatus.OPEN) {
                request.status = RemovalRequestStatus.RESOLVING
                request.candidateCache = null
                request.evidenceCleanupState = RemovalEvidenceCleanupState.PENDING
                request.evidenceCleanupNextAttemptAt = now
                changed = true
            } else if (!decisioned && request.status in setOf(RemovalRequestStatus.RESOLVING, RemovalRequestStatus.RESOLVED)) {
                request.status = RemovalRequestStatus.OPEN
                request.resolvedAt = null
                changed = true
            }
            if (changed) {
                val version = request.version
                request.updatedAt = now
                items.flush(); requests.flush()
                if (request.version == version) requests.advanceVersion(requestId, now)
                publisher.publishEvent(removalChangedEvent(requestId, null, now))
            }
            decisioned && request.status == RemovalRequestStatus.RESOLVING
        }!!
        if (purge) evidence.purgeResolvedEvidence(requestId)
        tx.executeWithoutResult {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            val selected = items.findByRequestIdOrderByIdAsc(requestId)
            photos.findAllByIdForUpdate(selected.map { it.photoIdSnapshot }.sorted())
            val proofCurrent = selected.filter { it.state == RemovalItemState.REMOVED }.all {
                jobs.findByPhotoId(it.photoIdSnapshot)?.let { job -> jobs.lock(job.id) }.confirms(it.photoIdSnapshot)
            }
            if (request.status == RemovalRequestStatus.RESOLVING && request.evidenceCleanupState == RemovalEvidenceCleanupState.DELETED &&
                proofCurrent && items.findPendingOwnerReviewItemIds(requestId).isEmpty() &&
                selected.none { it.state in setOf(RemovalItemState.CANDIDATE, RemovalItemState.VERIFIED, RemovalItemState.REMOVAL_AUTHORIZED) }) {
                request.status = RemovalRequestStatus.RESOLVED
                request.resolvedAt = clock.instant()
                request.updatedAt = clock.instant()
                history.record(RemovalHistoryEvent(requestId, actorType = RemovalActorType.SYSTEM, action = RemovalHistoryActions.RESOLVED,
                    fromState = "RESOLVING", toState = "RESOLVED", occurredAt = request.resolvedAt!!,
                    commandId = UUID.nameUUIDFromBytes("removal:resolved:$requestId:${request.version}".toByteArray())))
                requests.flush()
                commands.sanitizeTerminalResults(requestId)
                publisher.publishEvent(removalChangedEvent(requestId, null, clock.instant()))
            }
        }
        reviews.findByRequestIdOrderByIdAsc(requestId).forEach { notifier.attempt(it.id) }
        return queries.adminDetail(UUID(0, 0), requestId)
    }

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
                if (replay.items.any { it.id !in owned }) removalNotFound()
                return replay.copy(items = replay.items.map { it.copy(photo = it.photo?.let { photo -> photo.copy(price = photo.price.setScale(2)) }) })
            }
            commands.findByCommandId(command.commandId)?.let { return@execute it.replayOwner() }
            transitions.ensureVersion(request, command.expectedVersion)
            if (request.status != RemovalRequestStatus.OPEN || commands.findByRequestIdAndStateIn(requestId,
                    listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN)).isNotEmpty()) removalConflict()
            if (command.itemIds.isEmpty() || command.itemIds.size > 100 || command.itemIds.toSet().size != command.itemIds.size)
                removalInvalid("Select one to 100 distinct items")
            val selected = command.itemIds.map { owned[it] ?: removalNotFound() }
            val pendingOwnerReviews = items.findPendingOwnerReviewItemIds(requestId).toSet()
            if (selected.any { !it.awaitingDecision(pendingOwnerReviews) }) removalConflict()
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
            if (request.status == RemovalRequestStatus.OPEN) return@execute reverifyOwner(adminId, request, command, hash)
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

    private fun reverifyOwner(adminId: UUID, request: PhotoRemovalRequest, command: VerifyRemovalCommand, hash: String): PhotoRemovalRequestDto {
        validateText(command.verificationMethod, 2000)
        validateText(command.verificationNote, 2000)
        if (command.selectedPhotos.isEmpty() || command.selectedPhotos.size > 100 ||
            command.selectedPhotos.map { it.photoId }.toSet().size != command.selectedPhotos.size ||
            commands.findByRequestIdAndStateIn(request.id, listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN)).isNotEmpty()) removalConflict()
        val selected = items.findByRequestIdOrderByIdAsc(request.id).associateBy { it.photoIdSnapshot }
        val current = photos.findAllByIdForUpdate(command.selectedPhotos.map { it.photoId }.sorted()).associateBy { it.id }
        val groups = reviews.findByRequestIdOrderByIdAsc(request.id).associateBy { it.id }
        command.selectedPhotos.forEach {
            validateText(it.note ?: "", 2000)
            val item = selected[it.photoId] ?: removalConflict()
            val photo = current[it.photoId] ?: removalConflict()
            if (item.state !in setOf(RemovalItemState.VERIFIED, RemovalItemState.RETAINED_RESTRICTED) ||
                photo.eventId != request.eventIdSnapshot || photo.removalState == RemovalState.REMOVED ||
                photo.photographerId == groups[item.reviewId]?.ownerIdSnapshot) removalConflict()
        }
        val now = clock.instant()
        if (commands.insertClaim(command.commandId, request.id, RemovalCommandOperation.VERIFY.name, adminId, hash, null, now) == 0)
            return commands.findByCommandId(command.commandId)!!.replay(adminId, request.id, RemovalCommandOperation.VERIFY, hash, mapper)
        val byOwner = groups.values.associateBy { it.ownerIdSnapshot }.toMutableMap()
        command.selectedPhotos.forEach { selection ->
            val item = selected.getValue(selection.photoId)
            val ownerId = current.getValue(selection.photoId).photographerId
            val oldReview = item.reviewId
            val review = ownerId?.let { id -> byOwner.getOrPut(id) {
                reviews.saveAndFlush(PhotoRemovalReview(requestId = request.id, ownerIdSnapshot = id, ownerId = id))
            } }
            review?.apply {
                notifiedAt = null; notifiedMessageId = null; deadline = null; overdue = false; overdueAt = null
                unavailableReason = null; notificationNextAttemptAt = now
            }
            item.reviewId = review?.id
            item.ownerResponse = null; item.ownerResponseReason = null; item.ownerRespondedAt = null
            item.verificationActorId = adminId; item.verifiedAt = now; item.verificationReason = selection.note
            items.flush(); reviews.flush()
            history.record(RemovalHistoryEvent(request.id, item.id, item.reviewId, item.photoIdSnapshot,
                RemovalActorType.ADMIN, adminId, RemovalHistoryActions.VERIFIED, item.state.name, item.state.name,
                "${command.verificationMethod}: ${command.verificationNote}; ${selection.note}", now, command.commandId,
                mapOf("kind" to "owner_reverification", "previousReviewId" to (oldReview?.toString() ?: "unassigned"),
                    "noticeVersion" to (command.expectedVersion + 1).toString())))
        }
        requests.advanceVersion(request.id, now)
        val result = queries.adminDetail(adminId, request.id)
        commands.findByCommandId(command.commandId)!!.apply { this.result = mapper.valueToTree(result); state = RemovalCommandState.SUCCEEDED }
        commands.flush()
        publisher.publishEvent(removalChangedEvent(request.id, adminId, now))
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
    companion object { private val log = org.slf4j.LoggerFactory.getLogger(PhotoRemovalService::class.java) }
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

internal fun PhotoErasureJob?.confirms(photoId: UUID) = this != null && this.photoId == photoId &&
    outcome == PhotoErasureOutcome.COMPLETE && completedAt != null
