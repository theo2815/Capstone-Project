package com.quickpitik.service.removal

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoAvailabilityService
import com.quickpitik.service.photos.PhotoErasureService
import com.quickpitik.service.orders.OrderPhotoRemovalService
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.dto.photos.*
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import org.springframework.http.HttpStatus
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
    private val properties: PhotoRemovalProperties,
    private val autoHide: RunnerAutoHidePolicy,
) {
    // Stateless over repositories this service already holds; withdraw builds its receipt inside its own transaction.
    private val runnerQueries = RunnerPhotoRemovalQueryService(requests, items, audit)

    @Transactional(propagation = Propagation.NEVER)
    fun decide(adminId: UUID, requestId: UUID, command: DecideRemovalCommand): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, command)
        val result = tx.execute {
            requireActiveAdmin(users, adminId)
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
            // RD-6 (Q2): on a runner case a photo with other people in it is removed only on its own.
            val removing = selected.zip(command.decisions).filter { it.second.decision == RemovalDecision.REMOVE }.map { it.first }
            if (request.source == RemovalRequestSource.RUNNER && removing.size > 1 &&
                removing.any { otherPeopleVisible(current[it.photoIdSnapshot]?.faceCount) }) removalInvalid("Decide photos with other people one at a time")
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
        attemptInRequest(result.items.filter { it.id in selectedIds }.mapNotNull { it.cleanupJobId })
        reconcileAffected(requestId, result.items.filter { it.id in selectedIds }.map { it.photoIdSnapshot })
        return result
    }

    @Transactional(propagation = Propagation.NEVER)
    fun retryCleanup(adminId: UUID, requestId: UUID, command: RetryRemovalCleanupCommand): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, command)
        val result = tx.execute {
            requireActiveAdmin(users, adminId)
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
                erasure.makeDue(job)
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
        attemptInRequest(result.items.filter { it.id in command.itemIds }.mapNotNull { it.cleanupJobId })
        reconcileAffected(requestId, result.items.filter { it.id in command.itemIds }.map { it.photoIdSnapshot })
        return result
    }

    @Transactional(propagation = Propagation.NEVER)
    fun supportUpdate(adminId: UUID, requestId: UUID, command: RemovalSupportUpdateCommand): RemovalHistoryDto {
        val hash = commandHash(mapper, command)
        return tx.execute {
            requireActiveAdmin(users, adminId)
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

    // ponytail: a count cap, not a wall-clock budget; PhotoErasureSweep finishes the rest within cleanupInterval.
    // Upgrade to a bounded executor with a timeout if provider stalls ever surface as 524s.
    private fun attemptInRequest(jobIds: List<UUID>) {
        jobIds.distinct().take(properties.inRequestCleanupJobs).forEach { job ->
            runCatching { erasure.attempt(job) }.onFailure { log.warn("Removal cleanup remains pending for job {} ({})", job, it.javaClass.simpleName) }
        }
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
            requireActiveAdmin(users, adminId)
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
            val confirmed = saveVerifiedItems(requestId, command.selectedPhotos.map { selection ->
                VerifiedPick(fresh.getValue(selection.photoId), selection.source, scores[selection.photoId], selection.note ?: command.verificationNote)
            }, adminId, now)
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
            publisher.publishEvent(removalChangedEvent(requestId, adminId, now))
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
            requireActiveAdmin(users, adminId)
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
            requireActiveAdmin(users, adminId)
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(command.commandId)?.let { return@execute it.replay(adminId, requestId, RemovalCommandOperation.CANCEL, hash, mapper) }
            transitions.ensureVersion(request, command.expectedVersion)
            validateText(command.reason, 2000)
            val selected = items.findByRequestIdOrderByIdAsc(requestId)
            requireClosable(request, selected)
            val now = clock.instant()
            if (commands.insertClaim(command.commandId, requestId, RemovalCommandOperation.CANCEL.name, adminId, hash, null, now) == 0) {
                return@execute commands.findByCommandId(command.commandId)!!.replay(adminId, requestId, RemovalCommandOperation.CANCEL, hash, mapper)
            }
            closeCase(request, selected, RemovalActorType.ADMIN, adminId, command.commandId, command.reason, null, now)
            publisher.publishEvent(removalChangedEvent(requestId, adminId, now))
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

    /**
     * RD-6: on an open case an admin promotes suggestions (CANDIDATE → VERIFIED, hidden, the owner told), dismisses them
     * (→ EXCLUDED, never hidden) or demotes an auto-hide (→ CANDIDATE, public again unless another case holds it).
     */
    @Transactional(propagation = Propagation.NEVER)
    fun reviewSuggestions(adminId: UUID, requestId: UUID, command: ReviewSuggestionsCommand): PhotoRemovalRequestDto {
        val hash = commandHash(mapper, command)
        val op = RemovalCommandOperation.REVIEW_SUGGESTIONS
        val result = tx.execute {
            requireActiveAdmin(users, adminId)
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(command.commandId)?.let { return@execute it.replay(adminId, requestId, op, hash, mapper) }
            transitions.ensureVersion(request, command.expectedVersion)
            val ids = command.promote + command.dismiss + command.demote
            if (ids.isEmpty() || ids.size > 100 || ids.toSet().size != ids.size) removalInvalid("Select one to 100 distinct items")
            if (request.status != RemovalRequestStatus.OPEN ||
                commands.findByRequestIdAndStateIn(requestId, listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN)).isNotEmpty()) removalConflict()
            val all = items.findByRequestIdOrderByIdAsc(requestId).associateBy { it.id }
            fun pick(list: List<UUID>, allowed: (PhotoRemovalItem) -> Boolean) = list.map { all[it]?.takeIf(allowed) ?: invalidTransition() }
            val promote = pick(command.promote) { it.state == RemovalItemState.CANDIDATE }
            val dismiss = pick(command.dismiss) { it.state == RemovalItemState.CANDIDATE }
            val demote = pick(command.demote) { it.autoHiddenIn(request) && it.adminDecision == null }
            // The checks verify and reverifyOwner make: the photo still exists, is in this event and is not removed.
            val fresh = photos.findAllByIdForUpdate(promote.mapNotNull { it.photoId }.sorted()).associateBy { it.id }
            promote.forEach { item ->
                val photo = item.photoId?.let { fresh[it] }
                if (photo == null || photo.eventId != request.eventIdSnapshot || photo.removalState == RemovalState.REMOVED) invalidTransition()
            }
            val now = clock.instant()
            if (commands.insertClaim(command.commandId, requestId, op.name, adminId, hash, null, now) == 0)
                return@execute commands.findByCommandId(command.commandId)!!.replay(adminId, requestId, op, hash, mapper)
            val byOwner = reviews.findByRequestIdOrderByIdAsc(requestId).associateBy { it.ownerIdSnapshot }.toMutableMap()
            promote.forEach { item ->
                val ownerId = fresh.getValue(item.photoId!!).photographerId
                val existing = ownerId?.let { byOwner[it] }
                val review = ownerId?.let { id -> existing ?: reviews.saveAndFlush(PhotoRemovalReview(requestId = requestId,
                    ownerIdSnapshot = id, ownerId = id)).also { byOwner[id] = it } }
                // A notified owner must be told again, with a new notice and a fresh response clock (reverifyOwner's reset).
                val renotify = existing != null && (existing.notifiedAt != null || existing.unavailableReason != null)
                if (renotify) existing!!.apply {
                    notifiedAt = null; notifiedMessageId = null; deadline = null; overdue = false; overdueAt = null
                    unavailableReason = null; notificationNextAttemptAt = now
                }
                item.state = RemovalItemState.VERIFIED
                item.reviewId = review?.id
                item.verificationActorId = adminId
                item.verifiedAt = now
                history.record(RemovalHistoryEvent(requestId, item.id, item.reviewId, item.photoIdSnapshot, RemovalActorType.ADMIN,
                    adminId, RemovalHistoryActions.VERIFIED, RemovalItemState.CANDIDATE.name, item.state.name, null, now, command.commandId,
                    if (renotify) mapOf("kind" to "owner_reverification", "noticeVersion" to (command.expectedVersion + 1).toString()) else emptyMap()))
            }
            dismiss.forEach { item ->
                item.state = RemovalItemState.EXCLUDED
                history.record(RemovalHistoryEvent(requestId, item.id, null, item.photoIdSnapshot, RemovalActorType.ADMIN, adminId,
                    RemovalHistoryActions.SUGGESTION_DISMISSED, RemovalItemState.CANDIDATE.name, item.state.name, null, now, command.commandId))
            }
            demote.forEach { item ->
                item.state = RemovalItemState.CANDIDATE
                item.reviewId = null // suggestions belong to no owner review (RD-4)
                history.record(RemovalHistoryEvent(requestId, item.id, null, item.photoIdSnapshot, RemovalActorType.ADMIN, adminId,
                    RemovalHistoryActions.SUGGESTED, RemovalItemState.VERIFIED.name, item.state.name, null, now, command.commandId,
                    mapOf("kind" to "demoted")))
            }
            items.flush()
            reviews.flush()
            requests.advanceVersion(requestId, now)
            // Conditional availability writer clears the persistence context after its update.
            (promote + demote).mapNotNull { it.photoId }.toSet().takeIf { it.isNotEmpty() }?.let { availability.refreshRemovalState(it) }
            val detail = queries.adminDetail(adminId, requestId)
            commands.findByCommandId(command.commandId)!!.apply { this.result = mapper.valueToTree(detail); state = RemovalCommandState.SUCCEEDED }
            commands.flush()
            publisher.publishEvent(removalChangedEvent(requestId, adminId, now))
            detail
        }!!
        result.reviews.filter { it.notifiedAt == null && it.unavailableReason == null }.forEach { notifier.attempt(it.id) }
        return result
    }

    /**
     * RD-5: the runner withdraws their own case until removal is authorized. Photos go public again unless another case
     * holds them; the case waits in RESOLVING (close reason WITHDRAWN, so the runner may file again) until the purge.
     */
    @Transactional(propagation = Propagation.NEVER)
    fun withdraw(runnerId: UUID, requestId: UUID, commandId: UUID, expectedVersion: Long): RunnerRemovalRequestDetailDto {
        val hash = commandHash(mapper, WithdrawRunnerRemovalCommand(expectedVersion))
        fun PhotoRemovalCommand.replayRunner(): RunnerRemovalRequestDetailDto {
            ensureIdentity(runnerId, requestId, RemovalCommandOperation.CANCEL, hash)
            if (state != RemovalCommandState.SUCCEEDED || result == null) removalConflict()
            return mapper.treeToValue(result, RunnerRemovalRequestDetailDto::class.java)
        }
        val replay = tx.execute {
            val request = requests.findForUpdate(requestId)
                ?.takeIf { it.requesterUserId == runnerId && it.source == RemovalRequestSource.RUNNER } ?: removalNotFound()
            commands.findByCommandId(commandId)?.let { return@execute it.replayRunner() }
            transitions.ensureVersion(request, expectedVersion)
            val selected = items.findByRequestIdOrderByIdAsc(requestId)
            requireClosable(request, selected)
            val now = clock.instant()
            if (commands.insertClaim(commandId, requestId, RemovalCommandOperation.CANCEL.name, runnerId, hash, null, now) == 0)
                return@execute commands.findByCommandId(commandId)!!.replayRunner()
            closeCase(request, selected, RemovalActorType.RUNNER, runnerId, commandId, null, RunnerRemovalCloseReason.WITHDRAWN, now)
            publisher.publishEvent(removalChangedEvent(requestId, runnerId, now))
            commands.findByCommandId(commandId)!!.apply {
                result = mapper.valueToTree(runnerQueries.detailOf(request))
                state = RemovalCommandState.SUCCEEDED
            }
            commands.flush()
            null
        }
        if (replay != null) return replay
        evidence.purgeResolvedEvidence(requestId)
        // The receipt is the withdraw's own result, before the purge moves the case to CANCELLED.
        return tx.execute { commands.findByCommandId(commandId)!!.replayRunner() }!!
    }

    /** Transaction A of a runner submit: the case and its evidence key are stored before any upload. */
    fun openRunnerDraft(command: RunnerDraftCommand): RunnerDraft {
        val key = PhotoRemovalEvidenceService.recordedEvidenceKey(command.requestId, command.creationCommandId)
        try {
            tx.executeWithoutResult {
                val now = clock.instant()
                val reason = RunnerRemovalReason.valueOf(command.reasonCode).label + (command.details?.let { ": $it" } ?: "")
                requests.saveAndFlush(PhotoRemovalRequest(id = command.requestId, eventId = command.event.id,
                    eventIdSnapshot = command.event.id, eventNameSnapshot = command.event.name, requesterUserId = command.runnerId,
                    requesterContact = command.runnerEmail, supportReference = "Runner request", requesterReason = reason,
                    publicToOwnerReason = "Privacy removal request", creatorId = command.runnerId,
                    creationCommandId = command.creationCommandId, createdAt = now, source = RemovalRequestSource.RUNNER,
                    reasonCode = command.reasonCode, consentVersion = command.consentVersion, consentAt = command.consentAt,
                    livenessSessionId = command.livenessSessionId, livenessBand = command.livenessBand,
                    livenessConfidence = command.livenessConfidence, evidenceContentType = command.evidenceContentType,
                    riskFlags = mutableListOf(RunnerRiskFlag.MATCHING_UNAVAILABLE.name), outstandingEvidenceKeys = listOf(key),
                    evidenceCleanupState = RemovalEvidenceCleanupState.PENDING,
                    evidenceCleanupNextAttemptAt = now.plus(properties.cleanupBackoff)))
                history.record(RemovalHistoryEvent(requestId = command.requestId, actorType = RemovalActorType.RUNNER,
                    actorId = command.runnerId, action = RemovalHistoryActions.CREATED, toState = RemovalRequestStatus.DRAFT.name,
                    occurredAt = now, commandId = command.creationCommandId))
            }
        } catch (error: DataIntegrityViolationException) {
            // uq_photo_removal_requests_runner_open or _liveness_session: nothing was stored.
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.REMOVAL_REQUEST_OPEN, "You already have an open request.")
        }
        return RunnerDraft(command.requestId, key)
    }

    /** Transaction B of a runner submit: install the evidence, then hide the auto band and suggest the rest. */
    @Transactional(propagation = Propagation.NEVER)
    fun completeRunnerCase(requestId: UUID, outcome: RunnerMatchOutcome): RunnerCaseResult {
        var purge = false
        val result = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            val key = PhotoRemovalEvidenceService.recordedEvidenceKey(requestId, request.creationCommandId)
            val existing = items.findByRequestIdOrderByIdAsc(requestId)
            val now = clock.instant()
            fun trackKey() {
                request.outstandingEvidenceKeys = (request.outstandingEvidenceKeys + key).distinct()
                request.evidenceCleanupState = RemovalEvidenceCleanupState.PENDING
                request.evidenceCleanupNextAttemptAt = now
            }
            // A withdraw, admin cancel or admin verify took the lock first: a late upload is never installed.
            if (request.status != RemovalRequestStatus.DRAFT || existing.isNotEmpty()) {
                trackKey()
                requests.flush()
                purge = true
                return@execute RunnerCaseResult(emptyList(), false, 0, 0)
            }
            if (key in request.outstandingEvidenceKeys) {
                request.outstandingEvidenceKeys = request.outstandingEvidenceKeys - key
                request.evidenceKey = key
                request.evidenceGeneration++
                request.evidenceCleanupState = if (request.outstandingEvidenceKeys.isEmpty()) RemovalEvidenceCleanupState.PRESENT else RemovalEvidenceCleanupState.PENDING
                request.evidenceCleanupNextAttemptAt = if (request.outstandingEvidenceKeys.isEmpty()) null else now
            } else trackKey() // The sweep already deleted it (R11): an upload that landed after the sweep is deleted too.
            request.updatedAt = now
            if (outcome.matchingUnavailable) {
                requests.flush()
                publisher.publishEvent(removalChangedEvent(requestId, null, now))
                return@execute RunnerCaseResult(emptyList(), false, 0, 0)
            }
            if (outcome.candidates.isEmpty()) {
                closeCase(request, existing, RemovalActorType.SYSTEM, null,
                    UUID.nameUUIDFromBytes("removal:runner:$requestId:no-matches".toByteArray()), "No matching photos",
                    RunnerRemovalCloseReason.NO_MATCHES, now)
                publisher.publishEvent(removalChangedEvent(requestId, null, now))
                purge = true
                return@execute RunnerCaseResult(emptyList(), true, 0, 0)
            }
            // Lock order: case row, then owner rows, then photo rows (each sorted), so the daily cap holds across runners.
            val bandOwners = outcome.candidates.filter { it.similarity >= properties.runner.autoHideMinSimilarity }
                .mapNotNull { it.ownerId }.toSortedSet()
            bandOwners.forEach { users.lockById(it) }
            val manilaMidnight = now.atZone(MANILA).toLocalDate().atStartOfDay(MANILA).toInstant()
            val hiddenToday = if (bandOwners.isEmpty()) emptyMap() else
                items.countRunnerAutoHidesSince(request.eventIdSnapshot, bandOwners, manilaMidnight).associate { it.ownerId to it.count }
            val decision = autoHide.decide(AutoHideInput(request.livenessBand!!, outcome.emailVerified, outcome.priorRejection,
                outcome.possiblyTruncated, outcome.candidates, hiddenToday))
            val locked = photos.findAllByIdForUpdate(decision.autoHide.map { it.photoId }.sorted()).associateBy { it.id }
            val hide = decision.autoHide.mapNotNull { c -> locked[c.photoId]?.takeIf { it.eventId == request.eventIdSnapshot &&
                it.removalState != RemovalState.REMOVED }?.let { VerifiedPick(it, RemovalMatchSource.SELFIE, c.similarity, AUTO_REASON) } }
            val hidden = saveVerifiedItems(requestId, hide, null, now)
            val suggestOwners = users.findAllById(decision.suggest.mapNotNull { it.ownerId }.toSet()).associate { it.id to it.name }
            val present = photos.findAllById(decision.suggest.map { it.photoId }).filter { it.eventId == request.eventIdSnapshot }.map { it.id }.toSet()
            val suggested = decision.suggest.filter { it.photoId in present }.map { c ->
                items.save(PhotoRemovalItem(requestId = requestId, photoIdSnapshot = c.photoId, photoId = c.photoId,
                    ownerIdSnapshot = c.ownerId, ownerNameSnapshot = suggestOwners[c.ownerId], state = RemovalItemState.CANDIDATE,
                    matchSource = RemovalMatchSource.SELFIE, matchScore = c.similarity))
            }
            items.flush()
            request.status = RemovalRequestStatus.OPEN
            request.riskFlags = decision.flags.map { it.name }.sorted().toMutableList()
            requests.flush()
            hidden.forEach { history.record(RemovalHistoryEvent(requestId, it.id, it.reviewId, it.photoIdSnapshot,
                RemovalActorType.SYSTEM, null, RemovalHistoryActions.VERIFIED, toState = it.state.name,
                reason = it.verificationReason, occurredAt = now, commandId = request.creationCommandId)) }
            suggested.forEach { history.record(RemovalHistoryEvent(requestId, it.id, null, it.photoIdSnapshot,
                RemovalActorType.SYSTEM, null, RemovalHistoryActions.SUGGESTED, toState = it.state.name,
                occurredAt = now, commandId = request.creationCommandId)) }
            // Conditional availability writer clears the persistence context after its update.
            availability.refreshRemovalState(hidden.map { it.photoIdSnapshot }.toSet())
            publisher.publishEvent(removalChangedEvent(requestId, null, now))
            RunnerCaseResult(hidden.mapNotNull { it.reviewId }.distinct(), false, hidden.size, suggested.size)
        }!!
        if (purge) evidence.purgeResolvedEvidence(requestId)
        result.newReviewIds.forEach { notifier.attempt(it) }
        return result
    }

    private data class VerifiedPick(val photo: Photo, val source: RemovalMatchSource, val score: Double?, val reason: String?)

    // One review per owner and one VERIFIED item per pick (admin verify and runner auto-hide). The caller flushes items.
    private fun saveVerifiedItems(requestId: UUID, picks: List<VerifiedPick>, actorId: UUID?, now: java.time.Instant): List<PhotoRemovalItem> {
        val ownerIds = picks.mapNotNull { it.photo.photographerId }.toSet()
        val owners = users.findAllById(ownerIds).associateBy { it.id }
        val groups = ownerIds.associateWith { ownerId ->
            reviews.save(PhotoRemovalReview(requestId = requestId, ownerIdSnapshot = ownerId, ownerId = owners[ownerId]?.id))
        }
        reviews.flush()
        return picks.map { pick ->
            val photo = pick.photo
            items.save(PhotoRemovalItem(requestId = requestId, photoIdSnapshot = photo.id, photoId = photo.id,
                ownerIdSnapshot = photo.photographerId, ownerNameSnapshot = owners[photo.photographerId]?.name,
                reviewId = groups[photo.photographerId]?.id, state = RemovalItemState.VERIFIED, matchSource = pick.source,
                matchScore = pick.score, verificationActorId = actorId, verifiedAt = now, verificationReason = pick.reason))
        }
    }

    private fun requireClosable(request: PhotoRemovalRequest, selected: List<PhotoRemovalItem>) {
        if (!removalClosable(request.status, selected.map { it.state })) removalConflict()
    }

    // The caller runs evidence.purgeResolvedEvidence after commit, which moves the case to CANCELLED.
    private fun closeCase(request: PhotoRemovalRequest, selected: List<PhotoRemovalItem>, actorType: RemovalActorType,
        actorId: UUID?, commandId: UUID, reason: String?, closeReason: RunnerRemovalCloseReason?, now: java.time.Instant) {
        val previous = request.status
        selected.filter { it.state in setOf(RemovalItemState.CANDIDATE, RemovalItemState.VERIFIED, RemovalItemState.RETAINED_RESTRICTED) }
            .forEach { it.state = RemovalItemState.EXCLUDED }
        items.flush()
        request.status = RemovalRequestStatus.RESOLVING
        request.closeReason = closeReason
        request.candidateCache = null
        request.updatedAt = now
        request.evidenceCleanupState = RemovalEvidenceCleanupState.PENDING
        request.evidenceCleanupNextAttemptAt = now
        requests.flush()
        // Availability's native update clears the persistence context; persist the case first.
        availability.refreshRemovalState(selected.mapNotNull { it.photoId }.toSet())
        history.record(RemovalHistoryEvent(requestId = request.id, actorType = actorType, actorId = actorId,
            action = RemovalHistoryActions.CANCEL_REQUESTED, fromState = previous.name, toState = request.status.name,
            reason = reason, occurredAt = now, commandId = commandId))
    }

    companion object {
        private val log = org.slf4j.LoggerFactory.getLogger(PhotoRemovalService::class.java)
        private val MANILA = java.time.ZoneId.of("Asia/Manila")
        private const val AUTO_REASON = "Automatic: live check passed and match at or above the auto-hide value"
    }
}

data class RunnerDraftCommand(
    val requestId: UUID, val runnerId: UUID, val runnerEmail: String, val event: Event, val creationCommandId: UUID,
    val reasonCode: String, val details: String?, val consentVersion: String, val consentAt: java.time.Instant,
    val livenessSessionId: String, val livenessBand: RunnerLivenessBand, val livenessConfidence: java.math.BigDecimal,
    val evidenceContentType: String,
)
data class RunnerDraft(val requestId: UUID, val evidenceKey: String)
data class RunnerMatchOutcome(
    val matchingUnavailable: Boolean, val candidates: List<CandidatePhoto>, val possiblyTruncated: Boolean,
    val emailVerified: Boolean, val priorRejection: Boolean,
)
data class RunnerCaseResult(val newReviewIds: List<UUID>, val closedNoMatches: Boolean, val hiddenCount: Int, val suggestedCount: Int)

internal fun invalidTransition(): Nothing =
    throw ApiException(HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION, "Cannot change this photo removal item")
internal fun validateText(value: String, max: Int) {
    if (value.isBlank() || value.length > max) removalInvalid("Photo removal field is empty or too long")
}
internal fun requireActiveAdmin(users: UserRepository, adminId: UUID) {
    val actor = users.findById(adminId).orElse(null)
    if (actor == null || actor.role != Role.ADMIN || actor.suspendedAt != null)
        throw ApiException(HttpStatus.FORBIDDEN, ErrorCodes.FORBIDDEN, "An active admin is required")
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
