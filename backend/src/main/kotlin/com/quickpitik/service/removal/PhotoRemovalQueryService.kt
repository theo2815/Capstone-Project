package com.quickpitik.service.removal

import com.quickpitik.common.*
import com.quickpitik.dto.removal.*
import com.quickpitik.dto.photos.PhotographerRef
import com.quickpitik.dto.photos.toDto
import com.quickpitik.config.StorageProperties
import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.storage.StorageService
import org.springframework.data.domain.Sort
import org.springframework.data.jpa.domain.Specification
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.util.UUID

@Service
@Transactional(readOnly = true)
class PhotoRemovalQueryService(
    private val requests: PhotoRemovalRequestRepository,
    private val items: PhotoRemovalItemRepository,
    private val reviews: PhotoRemovalReviewRepository,
    private val history: PhotoRemovalHistoryRepository,
    private val photos: PhotoRepository,
    private val users: UserRepository,
    private val commands: PhotoRemovalCommandRepository,
    private val evidence: PhotoRemovalEvidenceService,
    private val search: PhotoSearchService,
    private val storage: StorageService,
    private val storageProperties: StorageProperties,
    private val tx: TransactionTemplate,
    private val mapper: ObjectMapper,
    private val clock: Clock,
) {

    @Transactional(propagation = Propagation.NEVER)
    fun photographerDetail(photographerId: UUID, requestId: UUID): PhotographerRemovalRequestDto {
        val snapshot = tx.execute { photographerSnapshot(photographerId, requestId) }!!
        return presentOwner(snapshot)
    }

    @Transactional(propagation = Propagation.NEVER)
    fun photographerList(photographerId: UUID, offset: Int?, limit: Int?, eventId: UUID? = null,
        status: RemovalRequestStatus? = null, needsAction: Boolean? = null): RemovalPage<PhotographerRemovalRequestDto> {
        val params = removalPagination(offset, limit)
        val page = tx.execute {
            val rows = requests.findForPhotographer(photographerId, eventId, status, needsAction, OffsetLimitPageable(params))
            RemovalPage(rows.content.map { photographerSnapshot(photographerId, it.id).dto.copy(items = emptyList()) },
                rows.totalElements, params.offset, params.limit)
        }!!
        return page
    }

    internal data class OwnerSnapshot(val dto: PhotographerRemovalRequestDto, val previewKeys: Map<UUID, String>)

    // Safe scalars and keys are captured while lazy photo fields are available. Signing happens after the transaction.
    internal fun photographerSnapshot(photographerId: UUID, requestId: UUID): OwnerSnapshot {
        // A later item snapshot may safely carry an older version; the reverse permits stale edits.
        val request = requests.findById(requestId).orElse(null) ?: removalNotFound()
        val owned = items.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(requestId, photographerId)
        if (owned.isEmpty()) removalNotFound()
        val current = photos.findAllById(owned.mapNotNull { it.photoId }).associateBy { it.id }
        val keys = mutableMapOf<UUID, String>()
        val selected = owned.map { item ->
            val photo = current[item.photoId]
            if (photo != null && photo.purgedAt == null && photo.deletionStartedAt == null && photo.removalState != RemovalState.REMOVED &&
                item.state !in setOf(RemovalItemState.REMOVAL_AUTHORIZED, RemovalItemState.REMOVED)) {
                (photo.thumbnailS3Key ?: photo.watermarkS3Key)?.let { keys[item.id] = it }
            }
            PhotographerRemovalItemDto(item.id, item.photoIdSnapshot, item.state, item.ownerResponse, item.ownerResponseReason,
                item.ownerRespondedAt, item.adminDecision, null, item.adminDecidedAt, item.retentionReviewAt,
                null, photo?.toDto(thumbnailUrlResolver = { null }), observedRemovalPhotoState(photo))
        }
        val ownReviews = reviews.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(requestId, photographerId).map {
            val group = owned.filter { item -> item.reviewId == it.id }
            PhotographerRemovalReviewDto(it.id, it.notifiedAt, it.deadline, it.overdue, it.overdueAt,
                RemovalResponseCounts(group.count { item -> item.awaitingDecision() && item.ownerResponse == null },
                    group.count { item -> item.ownerResponse == PhotographerRemovalResponse.AGREE },
                    group.count { item -> item.ownerResponse == PhotographerRemovalResponse.OBJECT },
                    group.count { item -> item.ownerResponse == PhotographerRemovalResponse.ALREADY_DELETED }))
        }
        return OwnerSnapshot(PhotographerRemovalRequestDto(request.id, RemovalEventDto(request.eventIdSnapshot, request.eventNameSnapshot),
            request.status, request.version, request.createdAt, request.updatedAt, request.resolvedAt, request.publicToOwnerReason,
            selected, ownReviews), keys)
    }

    private fun presentOwner(snapshot: OwnerSnapshot): PhotographerRemovalRequestDto {
        try {
            return snapshot.dto.copy(items = snapshot.dto.items.map { item ->
                val key = snapshot.previewKeys[item.id]
                val preview = if (key != null && storage.exists(key)) storage.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail) else null
                item.copy(photo = item.photo?.copy(imageUrl = preview))
            })
        } catch (error: Exception) {
            throw ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.UPLOAD_STORAGE_UNAVAILABLE, "Photo review storage is unavailable")
        }
    }

    @Transactional(propagation = Propagation.NEVER)
    fun photos(adminId: UUID, requestId: UUID, bib: String?, offset: Int?, limit: Int?): RemovalPage<RemovalPhotoDto> {
        val params = removalPagination(offset, limit)
        val rows = tx.execute {
            val request = requests.findById(requestId).orElse(null) ?: removalNotFound()
            val normalized = bib.orEmpty().removePrefix("B-").removePrefix("b-").trim().uppercase().filter { it.isLetterOrDigit() }
            val page = photos.searchForRemovalReview(request.eventIdSnapshot, normalized, OffsetLimitPageable(params))
            reviewPhotos(requestId, page.content) to page.totalElements
        }!!
        return RemovalPage(rows.first.map(::present), rows.second, params.offset, params.limit)
    }

    @Transactional(propagation = Propagation.NEVER)
    fun matches(adminId: UUID, requestId: UUID, command: MatchRemovalCommand): RemovalMatchResult {
        val hash = commandHash(mapper, command)
        val replay = tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(command.commandId)?.let { return@execute it.replayMatch(adminId, requestId, hash) }
            ensureCanMatch(request, command.expectedVersion)
            if (request.evidenceKey == null) removalNotFound()
            null
        }
        if (replay != null) return replay
        val sample = evidence.stream(adminId, requestId)
        val image = sample.stream.use {
            if (sample.requestVersion != command.expectedVersion) removalConflict()
            val bytes = try { it.readAllBytes() } catch (error: Exception) {
                throw ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.UPLOAD_STORAGE_UNAVAILABLE, "Case evidence storage is unavailable")
            }
            SearchImage(bytes, sample.contentType, "case-selfie")
        }
        val candidates = search.findFaceCandidates(sample.eventId, image)
        if (candidates.noFaces) throw AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NO_FACES", "No face detected in case evidence")
        val scores = candidates.matches.groupBy { it.personId }.mapValues { (_, matches) -> matches.maxOf { it.similarity } }
        val discovered = tx.execute {
            val found = if (scores.isEmpty()) emptyList() else photos.findRemovalReviewCandidates(sample.eventId, scores.keys)
            reviewPhotos(requestId, found, scores)
        }!!.map(::present)
        return tx.execute {
            val request = requests.findForUpdate(requestId) ?: removalNotFound()
            commands.findByCommandId(command.commandId)?.let { return@execute it.replayMatch(adminId, requestId, hash) }
            ensureCanMatch(request, command.expectedVersion)
            if (request.evidenceGeneration != sample.evidenceGeneration || request.evidenceKey != sample.evidenceKey) removalConflict()
            if (commands.insertClaim(command.commandId, requestId, RemovalCommandOperation.MATCH_CANDIDATES.name,
                    adminId, hash, null, clock.instant()) == 0) {
                return@execute commands.findByCommandId(command.commandId)!!.replayMatch(adminId, requestId, hash)
            }
            // One request flush advances @Version once. The cache names that committed version.
            request.candidateCache = RemovalCandidateCache(request.version + 1, sample.evidenceGeneration,
                discovered.map { RemovalCachedCandidate(it.photo.id, RemovalMatchSource.SELFIE, it.matchScore) })
            request.updatedAt = clock.instant()
            requests.flush()
            val result = RemovalMatchResult(discovered, candidates.candidateLimit, candidates.possiblyTruncated,
                request.version, sample.evidenceGeneration)
            val receipt = commands.findByCommandId(command.commandId)!!
            receipt.result = mapper.valueToTree(result)
            receipt.state = RemovalCommandState.SUCCEEDED
            commands.flush()
            result
        }!!
    }

    private fun ensureCanMatch(request: PhotoRemovalRequest, expectedVersion: Long) {
        if (expectedVersion < 0 || request.version != expectedVersion ||
            request.status !in setOf(RemovalRequestStatus.DRAFT, RemovalRequestStatus.OPEN) ||
            commands.findByRequestIdAndStateIn(request.id, listOf(RemovalCommandState.PENDING, RemovalCommandState.UNKNOWN)).isNotEmpty()) removalConflict()
    }

    private fun PhotoRemovalCommand.replayMatch(adminId: UUID, requestId: UUID, hash: String): RemovalMatchResult {
        ensureIdentity(adminId, requestId, RemovalCommandOperation.MATCH_CANDIDATES, hash)
        if (state != RemovalCommandState.SUCCEEDED || result == null) removalConflict()
        val replay = mapper.treeToValue(result, RemovalMatchResult::class.java)
        // JSONB normalizes whole numbers; restore the existing price column's monetary scale.
        return replay.copy(items = replay.items.map { it.copy(photo = it.photo.copy(price = it.photo.price.setScale(2))) })
    }

    private data class ReviewPhoto(val dto: RemovalPhotoDto, val originalKey: String?, val previewKey: String?)

    private fun reviewPhotos(requestId: UUID, found: List<Photo>, scores: Map<String, Double> = emptyMap()): List<ReviewPhoto> {
        val owners = users.findAllById(found.mapNotNull { it.photographerId }.toSet()).associate { it.id to it.name }
        val selected = items.findByRequestIdOrderByIdAsc(requestId).associateBy { it.photoIdSnapshot }
        return found.map { photo ->
            val score = photo.facePersons.mapNotNull { scores[it.aiPersonId] }.maxOrNull()
            val dto = photo.toDto(thumbnailUrlResolver = { null },
                photographerResolver = { it.photographerId?.let { id -> PhotographerRef(null, owners[id]) } })
            val retained = photo.purgedAt == null && photo.removalState != RemovalState.REMOVED && photo.deletionStartedAt == null
            ReviewPhoto(RemovalPhotoDto(dto, photo.photographerId, owners[photo.photographerId], selected[photo.id]?.id,
                photo.removalState, photo.deletedAt != null, false, score, if (score != null) RemovalMatchSource.SELFIE else null),
                photo.s3Key.takeIf { retained && it.isNotBlank() },
                (photo.thumbnailS3Key ?: photo.watermarkS3Key).takeIf { retained })
        }
    }

    private fun present(row: ReviewPhoto): RemovalPhotoDto {
        try {
            val available = row.originalKey?.let(storage::exists) == true
            val preview = if (available && row.previewKey?.let(storage::exists) == true)
                storage.presignedGetUrl(row.previewKey, storageProperties.presignedTtl.thumbnail) else null
            return row.dto.copy(photo = row.dto.photo.copy(imageUrl = preview), bytesAvailable = available)
        } catch (error: Exception) {
            throw ApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.UPLOAD_STORAGE_UNAVAILABLE, "Photo review storage is unavailable")
        }
    }
    fun adminDetail(adminId: UUID, requestId: UUID): PhotoRemovalRequestDto {
        val request = requests.findById(requestId).orElse(null) ?: removalNotFound()
        val selected = items.findByRequestIdOrderByIdAsc(requestId)
        return request.toAdminDto(selected, reviews.findByRequestIdOrderByIdAsc(requestId))
    }

    fun list(adminId: UUID, eventId: UUID?, status: RemovalRequestStatus?, needsAction: Boolean?, offset: Int?, limit: Int?): RemovalPage<PhotoRemovalRequestDto> {
        val params = removalPagination(offset, limit)
        val filter = Specification<PhotoRemovalRequest> { root, query, cb ->
            val predicates = mutableListOf<jakarta.persistence.criteria.Predicate>()
            eventId?.let { predicates += cb.equal(root.get<UUID>("eventIdSnapshot"), it) }
            status?.let { predicates += cb.equal(root.get<RemovalRequestStatus>("status"), it) }
            needsAction?.let {
                val retained = query!!.subquery(UUID::class.java)
                val item = retained.from(PhotoRemovalItem::class.java)
                retained.select(item.get("id")).where(cb.equal(item.get<UUID>("requestId"), root.get<UUID>("id")),
                    cb.equal(item.get<RemovalItemState>("state"), RemovalItemState.RETAINED_RESTRICTED),
                    cb.lessThanOrEqualTo(item.get("retentionReviewAt"), clock.instant()))
                val active = cb.or(root.get<RemovalRequestStatus>("status").`in`(RemovalRequestStatus.DRAFT, RemovalRequestStatus.OPEN, RemovalRequestStatus.RESOLVING),
                    cb.and(cb.equal(root.get<RemovalRequestStatus>("status"), RemovalRequestStatus.RESOLVED), cb.exists(retained)))
                predicates += if (it) active else cb.not(active)
            }
            cb.and(*predicates.toTypedArray())
        }
        val page = requests.findAll(filter, OffsetLimitPageable(params, Sort.by(Sort.Direction.DESC, "createdAt").and(Sort.by("id"))))
        return RemovalPage(page.content.map { it.toAdminDto(items.findByRequestIdOrderByIdAsc(it.id), emptyList()).copy(items = emptyList()) },
            page.totalElements, params.offset, params.limit)
    }

    fun history(adminId: UUID, requestId: UUID, offset: Int?, limit: Int?): RemovalPage<RemovalHistoryDto> {
        if (!requests.existsById(requestId)) removalNotFound()
        val params = removalPagination(offset, limit)
        val rows = history.findByRequestIdOrderByOccurredAtAscIdAsc(requestId, OffsetLimitPageable(params))
        return RemovalPage(rows.map { RemovalHistoryDto(it.id, it.requestId, it.itemId, it.reviewId, it.photoIdSnapshot,
            it.actorType, it.actorIdSnapshot, it.action, it.fromState, it.toState, it.reason, it.occurredAt, it.commandId, it.metadata) },
            history.countByRequestId(requestId), params.offset, params.limit)
    }
}

internal fun removalPagination(offset: Int?, limit: Int?) = PaginationParams((offset ?: 0).coerceAtLeast(0), (limit ?: 50).coerceIn(1, 100))
internal fun observedRemovalPhotoState(photo: Photo?) = when {
    photo == null -> "ROW_ABSENT_UNCONFIRMED"
    photo.purgedAt != null -> "PURGED_UNCONFIRMED"
    photo.deletionStartedAt != null -> "DELETION_PENDING"
    photo.deletedAt != null -> "TRASHED"
    else -> "PRESENT"
}
internal fun removalNotFound(): Nothing = throw ApiException(HttpStatus.NOT_FOUND, ErrorCodes.NOT_FOUND, "Photo removal request not found")
internal fun removalConflict(): Nothing = throw ApiException(HttpStatus.CONFLICT, ErrorCodes.CONFLICT, "Photo removal command changed or is still in progress")
internal fun removalInvalid(message: String): Nothing = throw ApiException(HttpStatus.BAD_REQUEST, ErrorCodes.VALIDATION_ERROR, message)

internal fun PhotoRemovalRequest.toAdminDto(items: List<PhotoRemovalItem>, reviews: List<PhotoRemovalReview>): PhotoRemovalRequestDto {
    fun count(state: RemovalItemState) = items.count { it.state == state }.toLong()
    return PhotoRemovalRequestDto(id, RemovalEventDto(eventIdSnapshot, eventNameSnapshot), status, version, createdAt, updatedAt,
        resolvedAt, RemovalPhaseCounts(count(RemovalItemState.CANDIDATE), count(RemovalItemState.VERIFIED), count(RemovalItemState.EXCLUDED),
            count(RemovalItemState.REMOVAL_AUTHORIZED), count(RemovalItemState.REMOVED), count(RemovalItemState.RETAINED), count(RemovalItemState.RETAINED_RESTRICTED)),
        requesterUserId, requesterContact, supportReference, requesterReason, publicToOwnerReason, creatorId, verifierId,
        verificationMethod, verificationNote, verifiedAt, evidenceKey != null, evidenceContentType, evidenceGeneration,
        evidenceCleanupState, outstandingEvidenceKeys.size,
        items.map { PhotoRemovalItemDto(it.id, it.photoIdSnapshot, it.ownerIdSnapshot, it.ownerNameSnapshot, it.reviewId, it.state,
            it.matchSource, it.matchScore, it.verificationActorId, it.verifiedAt, it.verificationReason, it.ownerResponse,
            it.ownerResponseReason, it.ownerRespondedAt, it.adminDecision, it.adminDecisionReason, it.adminDecisionActorId,
            it.adminDecidedAt, it.retentionReviewAt, it.cleanupJobId, null) },
        reviews.map { PhotoRemovalReviewDto(it.id, it.ownerIdSnapshot, it.notifiedMessageId, it.notifiedAt, it.deadline,
            it.overdue, it.overdueAt, it.unavailableReason) })
}
