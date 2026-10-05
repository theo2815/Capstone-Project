package com.quickpitik.dto.removal

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.entity.*
import jakarta.validation.Valid
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.PositiveOrZero
import jakarta.validation.constraints.Size
import java.time.Instant
import java.util.UUID

typealias RemovalPage<T> = PaginatedResponse<T>

data class RemovalEventDto(val id: UUID, val name: String)
data class RemovalPhaseCounts(
    val candidate: Long = 0, val verified: Long = 0, val excluded: Long = 0,
    val removalAuthorized: Long = 0, val removed: Long = 0, val retained: Long = 0, val retainedRestricted: Long = 0,
)
data class RemovalPhotoDto(
    val photo: PhotoDto,
    val photographerId: UUID?,
    val photographerName: String?,
    val itemId: UUID? = null,
    val removalState: RemovalState,
    val trashed: Boolean,
    val bytesAvailable: Boolean,
    val matchScore: Double? = null,
    val matchSource: RemovalMatchSource? = null,
)
data class PhotoRemovalItemDto(
    val id: UUID, val photoIdSnapshot: UUID, val ownerIdSnapshot: UUID?, val ownerNameSnapshot: String?,
    val reviewId: UUID?, val state: RemovalItemState, val matchSource: RemovalMatchSource, val matchScore: Double?,
    val verificationActorId: UUID?, val verifiedAt: Instant?, val verificationReason: String?,
    val ownerResponse: PhotographerRemovalResponse?, val ownerResponseReason: String?, val ownerRespondedAt: Instant?,
    val adminDecision: RemovalDecision?, val adminDecisionReason: String?, val adminDecisionActorId: UUID?,
    val adminDecidedAt: Instant?, val retentionReviewAt: Instant?, val cleanupJobId: UUID?, val cleanupState: String?,
    val photo: RemovalPhotoDto? = null,
)
data class PhotoRemovalReviewDto(
    val id: UUID, val ownerIdSnapshot: UUID, val notifiedMessageId: UUID?, val notifiedAt: Instant?,
    val deadline: Instant?, val overdue: Boolean, val overdueAt: Instant?, val unavailableReason: String?,
)
data class PhotoRemovalRequestDto(
    val id: UUID, val event: RemovalEventDto, val status: RemovalRequestStatus, val version: Long,
    val createdAt: Instant, val updatedAt: Instant, val resolvedAt: Instant?, val phaseCounts: RemovalPhaseCounts,
    val requesterUserId: UUID?, val requesterContact: String, val supportReference: String,
    val requesterReason: String, val publicToOwnerReason: String, val creatorId: UUID, val verifierId: UUID?,
    val verificationMethod: String?, val verificationNote: String?, val verifiedAt: Instant?,
    val evidencePresent: Boolean, val evidenceContentType: String?, val evidenceGeneration: Long,
    val evidenceCleanupState: RemovalEvidenceCleanupState, val outstandingEvidenceCount: Int,
    val items: List<PhotoRemovalItemDto> = emptyList(), val reviews: List<PhotoRemovalReviewDto> = emptyList(),
)

// Separate shapes keep intake, evidence and support verification out of owner reads.
data class PhotographerRemovalItemDto(
    val id: UUID, val photoIdSnapshot: UUID, val state: RemovalItemState,
    val response: PhotographerRemovalResponse?, val responseReason: String?, val respondedAt: Instant?,
    val decision: RemovalDecision?, val decisionReason: String?, val decidedAt: Instant?,
    val retentionReviewAt: Instant?, val cleanupState: String?, val photo: PhotoDto? = null,
)
data class PhotographerRemovalReviewDto(
    val id: UUID, val notifiedAt: Instant?, val deadline: Instant?, val overdue: Boolean, val overdueAt: Instant?,
)
data class PhotographerRemovalRequestDto(
    val id: UUID, val event: RemovalEventDto, val status: RemovalRequestStatus, val version: Long,
    val createdAt: Instant, val updatedAt: Instant, val resolvedAt: Instant? = null, val sanitizedReason: String,
    val items: List<PhotographerRemovalItemDto>, val reviews: List<PhotographerRemovalReviewDto>,
)
data class RemovalHistoryDto(
    val id: UUID, val requestId: UUID, val itemId: UUID?, val reviewId: UUID?, val photoIdSnapshot: UUID?,
    val actorType: RemovalActorType, val actorIdSnapshot: UUID?, val action: String, val fromState: String?,
    val toState: String?, val reason: String?, val occurredAt: Instant, val commandId: UUID, val metadata: Map<String, String>,
)
data class RemovalMatchResult(val items: List<RemovalPhotoDto>, val candidateLimit: Int, val possiblyTruncated: Boolean)

data class CreateRemovalRequestCommand(
    val eventId: UUID,
    @field:NotBlank val supportReference: String,
    @field:NotBlank val requesterContact: String,
    val requesterUserId: UUID? = null,
    @field:NotBlank val reason: String,
    val commandId: UUID,
)
data class VerifyRemovalSelection(val photoId: UUID, val source: RemovalMatchSource, val note: String? = null)
data class VerifyRemovalCommand(
    val commandId: UUID,
    @field:PositiveOrZero val expectedVersion: Long,
    @field:NotBlank val verificationMethod: String,
    @field:NotBlank val verificationNote: String,
    @field:Valid @field:Size(min = 1, max = 100) val selectedPhotos: List<VerifyRemovalSelection>,
)
data class RespondToRemovalCommand(
    val commandId: UUID,
    @field:PositiveOrZero val expectedVersion: Long,
    @field:Size(min = 1, max = 100) val itemIds: List<UUID>,
    val response: PhotographerRemovalResponse,
    val reason: String? = null,
)
data class RemovalItemDecision(
    val itemId: UUID, val decision: RemovalDecision,
    @field:NotBlank val reason: String,
    val retentionReviewAt: Instant? = null,
)
data class DecideRemovalCommand(
    val commandId: UUID,
    @field:PositiveOrZero val expectedVersion: Long,
    @field:Valid @field:Size(min = 1, max = 100) val decisions: List<RemovalItemDecision>,
    val urgentReason: String? = null,
)
