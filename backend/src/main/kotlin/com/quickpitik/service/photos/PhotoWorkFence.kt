package com.quickpitik.service.photos

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.*
import com.quickpitik.entity.RemovalState
import com.quickpitik.repository.PhotoRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.OffsetDateTime
import java.util.UUID

enum class PhotoWorkKind { WATERMARK, INDEXING }
data class PhotoWorkClaim(val photoId: UUID, val generation: Long, val token: UUID,
    val expiresAt: Instant, val kind: PhotoWorkKind)

/** A local expiry refuses settlement, but never proves a remote writer has ended.
 * Unfinished tokens cannot be stolen. TASK 9/10 must reconcile unknown operations.
 */
@Service
@Transactional
class PhotoWorkFence(
    private val photos: PhotoRepository,
    private val storage: StorageProperties,
    private val ai: AiApiProperties,
    private val provider: AiProperties,
    private val rekognition: RekognitionProperties,
    private val json: ObjectMapper,
    private val clock: Clock,
) {
    fun claim(photoId: UUID, kind: PhotoWorkKind): PhotoWorkClaim? {
        val photo = photos.findAllByIdForUpdate(listOf(photoId)).singleOrNull() ?: return null
        if (photo.removalState != RemovalState.NONE || photo.deletedAt != null || photo.deletionStartedAt != null) return null
        val token = if (kind == PhotoWorkKind.WATERMARK) photo.watermarkWorkToken else photo.indexingWorkToken
        if (token != null) return null
        if (photos.hasUnfinishedWork(photoId, kind.name)) return null
        val expires = OffsetDateTime.now(clock).plus(budget(kind))
        val claim = PhotoWorkClaim(photo.id, photo.workGeneration, UUID.randomUUID(), expires.toInstant(), kind)
        if (photos.claimWork(photo.id, kind.name, claim.token, expires) != 1) return null
        photos.recordWork(claim.token, photo.id, photo.eventId, claim.generation, kind.name,
            if (kind == PhotoWorkKind.WATERMARK) "storage" else provider.provider.name.lowercase(), expires,
            if (kind == PhotoWorkKind.WATERMARK) "events/${photo.eventId}/photos/${photo.id}/watermark.jpg" else null,
            if (kind == PhotoWorkKind.INDEXING) photo.id.toString() else null,
            photo.indexedProvider,
            json.writeValueAsString(photo.facePersons.map { it.aiPersonId }))
        return claim
    }

    // A caller settling results calls this inside its write transaction: the
    // photo lock then remains held until that transaction commits.
    fun isCurrent(claim: PhotoWorkClaim): Boolean {
        val photo = photos.findAllByIdForUpdate(listOf(claim.photoId)).singleOrNull() ?: return false
        val token = if (claim.kind == PhotoWorkKind.WATERMARK) photo.watermarkWorkToken else photo.indexingWorkToken
        val expiry = if (claim.kind == PhotoWorkKind.WATERMARK) photo.watermarkWorkExpiresAt else photo.indexingWorkExpiresAt
        return token == claim.token && photo.workGeneration == claim.generation &&
            expiry?.toInstant()?.isAfter(clock.instant()) == true && photo.removalState == RemovalState.NONE &&
            photo.deletedAt == null && photo.deletionStartedAt == null
    }

    // Only callers with proof that ALL their external operations ended may finish.
    fun finish(claim: PhotoWorkClaim) {
        photos.findAllByIdForUpdate(listOf(claim.photoId))
        photos.finishWorkOutput(claim.token, OffsetDateTime.now(clock))
        photos.finishWork(claim.photoId, claim.token)
    }

    fun recordPerson(claim: PhotoWorkClaim, personId: String) {
        photos.findAllByIdForUpdate(listOf(claim.photoId))
        photos.recordWorkPersons(claim.token, json.writeValueAsString(listOf(personId)))
    }

    fun attributeBatch(claim: PhotoWorkClaim, batchId: UUID) { photos.attributeWorkBatch(claim.token, batchId) }
    fun recordJob(claim: PhotoWorkClaim, kind: String, jobId: String) {
        // Erasure callbacks invalidate the job and tombstone in that order.
        // Take the photo lock first, matching erasure claim/settlement.
        photos.findAllByIdForUpdate(listOf(claim.photoId))
        photos.recordWorkJob(claim.token, json.writeValueAsString(mapOf(kind to jobId)))
    }

    // Observation of a nonterminal remote job permits another local polling
    // window, never revival of an expired/restricted generation or completion.
    fun renewBatchClaim(claim: PhotoWorkClaim) {
        if (claim.kind != PhotoWorkKind.INDEXING || !isCurrent(claim)) return
        photos.renewIndexingWork(claim.photoId, claim.token, OffsetDateTime.now(clock).plus(budget(claim.kind)))
    }

    @Transactional(readOnly = true)
    fun forBatch(batchId: UUID): List<PhotoWorkClaim> = photos.workForBatch(batchId).map {
        PhotoWorkClaim(it.photoId, it.generation, it.token, it.expiresAt, PhotoWorkKind.valueOf(it.kind))
    }

    private fun budget(kind: PhotoWorkKind): Duration {
        if (kind == PhotoWorkKind.WATERMARK) return storage.operationTimeout.multipliedBy(3)
        val inference = if (provider.provider == AiProvider.REKOGNITION) rekognition.operationTimeout.multipliedBy(3)
        else {
            val call = ai.connectTimeout.multipliedBy(2).plus(ai.readTimeout)
            val backoff = (1..ai.maxRetries).sumOf { ai.backoffBaseMillis shl (it - 1).coerceAtMost(6) }
            call.multipliedBy((ai.maxRetries + 1).toLong()).plusMillis(backoff)
        }
        val synchronous = storage.operationTimeout.plus(inference)
        return if (ai.indexingMode == IndexingMode.BATCH)
            synchronous.plusSeconds(ai.webhook.pollGraceSeconds).plusMillis(ai.webhook.pollIntervalMs)
        else synchronous
    }
}
