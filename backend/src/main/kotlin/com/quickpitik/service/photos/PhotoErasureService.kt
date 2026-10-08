package com.quickpitik.service.photos

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photos.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.*
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID

@Service
class PhotoErasureService(
    private val photos: PhotoRepository,
    private val jobs: PhotoErasureJobRepository,
    private val intents: PhotoUploadIntentRepository,
    private val items: PhotoRemovalItemRepository,
    private val orderItems: OrderItemRepository,
    private val grants: DownloadGrantRepository,
    private val coverage: EventPhotographerRepository,
    private val events: EventRepository,
    private val metadata: PhotoMetadataRepository,
    private val storage: StorageService,
    private val faces: PhotoFaceErasureService,
    private val properties: PhotoRemovalProperties,
    private val storageProperties: StorageProperties,
    private val tx: TransactionTemplate,
    private val json: ObjectMapper,
    private val clock: Clock,
) {
    fun request(command: PhotoErasureCommand): UUID {
        if (command.reason.isBlank() || command.reason.length > 2000) invalid()
        if (command.purpose == PhotoErasurePurpose.PRIVACY_REMOVAL &&
            (command.authorizedActorType != RemovalActorType.ADMIN || command.itemId == null || command.requestId == null)) invalid()
        return tx.execute {
            val photo = photos.findAllByIdForUpdate(listOf(command.photoId)).singleOrNull()
            val existing = jobs.findByPhotoId(command.photoId)?.let { jobs.lock(it.id) }
            authorize(command, photo, existing)
            if (existing != null) {
                if (command.purpose == PhotoErasurePurpose.PRIVACY_REMOVAL && existing.authorizations.none { it.itemId == command.itemId }) {
                    existing.authorizations = existing.authorizations + privacyAuthorization(command)
                    existing.retainTombstone = true
                    existing.revision++
                    jobs.saveAndFlush(existing)
                }
                return@execute existing.id
            }
            photo ?: throw NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
            val now = clock.instant()
            val receipts = intents.findByPhotoIdOrResultPhotoId(photo.id, photo.id)
            val manifest = ErasureManifest(
                (listOfNotNull(photo.s3Key, photo.thumbnailS3Key, photo.watermarkS3Key,
                    "events/${photo.eventId}/photos/${photo.id}/watermark.jpg") + photos.erasureWorkKeys(photo.id) +
                    receipts.flatMap { listOf(it.stagingKey, it.originalKey) }).filter { it.isNotBlank() }.distinct(),
                receipts.map { snapshotIntent(it, now) },
                photo.facePersons.isEmpty() && photo.indexingAttempts == 0 && photo.indexedProvider == null &&
                    photos.faceWorkForPhoto(photo.eventId, photo.id).isEmpty() && !photos.hasUnfinishedWork(photo.id, "INDEXING"),
            )
            // Snapshot legacy face pointers into the existing independent ledger
            // before a later missing photo row could destroy their attribution.
            photo.facePersons.groupBy { it.provider }.forEach { (provider, persons) ->
                val token = UUID.randomUUID()
                photos.recordWork(token, photo.id, photo.eventId, photo.workGeneration, "INDEXING", provider.name.lowercase(),
                    OffsetDateTime.now(clock), null, photo.id.toString(), null, "[]")
                photos.recordWorkPersons(token, json.writeValueAsString(persons.map { it.aiPersonId }))
                photos.finishWorkOutput(token, OffsetDateTime.now(clock))
            }
            val job = PhotoErasureJob(photoId = photo.id, eventId = photo.eventId, ownerId = photo.photographerId,
                purpose = command.purpose, authorizedActorType = command.authorizedActorType,
                authorizedActorId = command.authorizedActorId, reason = command.reason.trim(),
                requestId = command.requestId, itemId = command.itemId, commandId = command.commandId,
                manifest = manifest, generation = photo.workGeneration, nextAttemptAt = now, createdAt = now)
            if (command.purpose == PhotoErasurePurpose.PRIVACY_REMOVAL) job.authorizations = listOf(privacyAuthorization(command))
            jobs.saveAndFlush(job)
            if (photos.beginErasure(photo.id, OffsetDateTime.now(clock)) == 1 && photo.deletedAt == null) {
                photo.photographerId?.let { coverage.adjustPhotoCount(photo.eventId, it, -1) }
                events.adjustPhotoCount(photo.eventId, -1)
            }
            job.id
        }!!
    }

    @Transactional(propagation = Propagation.MANDATORY)
    fun recordUploadIntent(photoId: UUID, intent: PhotoUploadIntent) {
        // Keep receipt association and cleanup invalidation under photo → job → receipt locks.
        photos.findAllByIdForUpdate(listOf(photoId))
        val job = jobs.findByPhotoId(photoId)?.let { jobs.lock(it.id) } ?: return
        if ((job.manifest.intents + job.additionalIntents).any { it.id == intent.photoId }) return
        val now = clock.instant()
        job.additionalIntents = job.additionalIntents + snapshotIntent(intent, now)
        job.revision++
        job.outcome = PhotoErasureOutcome.PENDING
        job.completedAt = null
        job.resourceOutcomes = emptyMap()
        job.nextAttemptAt = now
        job.updatedAt = now
        jobs.saveAndFlush(job)
        photos.clearErasureConfirmation(photoId)
    }

    private fun snapshotIntent(intent: PhotoUploadIntent, now: java.time.Instant) =
        ErasureIntent(intent.photoId, intent.stagingKey, intent.originalKey, maxOf(intent.putExpiresAt.toInstant(),
            intent.cleanupAfter?.toInstant() ?: now,
            (intent.cleanedAt?.toInstant() ?: maxOf(now, intent.putExpiresAt.toInstant()).plus(storageProperties.directUpload.cleanupDelay))
                .plus(storageProperties.directUpload.cleanupRecheckWindow)))

    private fun authorize(command: PhotoErasureCommand, photo: Photo?, existing: PhotoErasureJob?) {
        if (command.purpose == PhotoErasurePurpose.PRIVACY_REMOVAL) {
            val item = items.findById(command.itemId!!).orElse(null) ?: invalid()
            if (item.requestId != command.requestId || item.photoIdSnapshot != command.photoId ||
                item.state !in setOf(RemovalItemState.REMOVAL_AUTHORIZED, RemovalItemState.REMOVED) ||
                item.adminDecision != RemovalDecision.REMOVE || item.adminDecisionActorId != command.authorizedActorId ||
                (photo != null && (item.photoId != photo.id || photo.removalState != RemovalState.REMOVED))) invalid()
        } else {
            if (command.authorizedActorType != RemovalActorType.PHOTOGRAPHER || command.requestId != null || command.itemId != null ||
                (photo?.photographerId ?: existing?.ownerId) != command.authorizedActorId ||
                (photo != null && (photo.deletedAt == null || photo.removalState != RemovalState.NONE)) ||
                existing?.retainTombstone == true) invalid()
            if (orderItems.existsByIdPhotoId(command.photoId) || grants.existsByIdPhotoId(command.photoId)) invalid()
        }
    }

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun attempt(jobId: UUID): PhotoErasureOutcome {
        val target = jobs.findById(jobId).orElseThrow { NotFoundException(message = "Erasure job not found") }
        val claim = tx.execute {
            photos.findAllByIdForUpdate(listOf(target.photoId))
            val job = jobs.lock(jobId) ?: return@execute null
            if (job.outcome == PhotoErasureOutcome.COMPLETE || job.leaseUntil?.isAfter(clock.instant()) == true) return@execute null
            job.leaseToken = UUID.randomUUID()
            job.leaseUntil = clock.instant().plus(storageProperties.operationTimeout.multipliedBy((job.manifest.keys.size * 2 + job.additionalIntents.size * 4 + 2).toLong()))
                .plus(properties.cleanupBackoff)
            job.attempts++
            job.executionActorType = RemovalActorType.SYSTEM
            job.executionActorId = null
            job.updatedAt = clock.instant()
            jobs.saveAndFlush(job)
        } ?: return if (jobs.findById(jobId).orElseThrow().outcome == PhotoErasureOutcome.COMPLETE)
            PhotoErasureOutcome.COMPLETE else PhotoErasureOutcome.PENDING

        val outcomes = linkedMapOf<String, String>()
        val receipts = claim.manifest.intents + claim.additionalIntents
        var faceOutcome = "PENDING"
        var failure: String? = null
        try {
            faceOutcome = if (claim.manifest.neverIndexed && photos.faceWorkForPhoto(claim.eventId, claim.photoId).isEmpty()) "ALREADY_ABSENT"
            else faces.erase(claim.eventId, claim.photoId).name
        } catch (_: Exception) { failure = "FACE_CLEANUP_REQUIRED" }
        var pending = photos.hasUnfinishedWork(claim.photoId, "ANY") ||
            photos.findById(claim.photoId).orElse(null)?.let { it.indexingWorkToken != null || it.watermarkWorkToken != null } == true
        val unresolvedUpload = receipts.any { saved ->
            val intent = intents.findById(saved.id).orElse(null)
            intent?.state in setOf(PhotoUploadIntentState.PENDING, PhotoUploadIntentState.VERIFYING) || intent?.leaseToken != null
        }
        pending = pending || unresolvedUpload
        if (!pending) {
            val keys = (claim.manifest.keys + claim.additionalIntents.flatMap { listOf(it.stagingKey, it.originalKey) } +
                photos.erasureWorkKeys(claim.photoId)).distinct()
            keys.forEach { key ->
                try {
                    if (photos.hasOtherBlobReference(claim.photoId, key)) {
                        outcomes[key] = "SHARED_REFERENCE"
                        failure = "SHARED_BLOB_REFERENCE"
                    } else {
                        storage.delete(key)
                        if (storage.exists(key)) { outcomes[key] = "RETRY_REQUIRED"; failure = "STORAGE_NOT_ABSENT" }
                        else outcomes[key] = "ABSENT"
                    }
                } catch (_: Exception) { outcomes[key] = "RETRY_REQUIRED"; failure = "STORAGE_CLEANUP_REQUIRED" }
            }
        }
        receipts.forEach { saved ->
            if (clock.instant().isBefore(saved.recheckUntil)) pending = true
        }
        return tx.execute {
            val photo = photos.findAllByIdForUpdate(listOf(claim.photoId)).singleOrNull()
            val job = jobs.lock(jobId) ?: return@execute PhotoErasureOutcome.PENDING
            if (job.leaseToken != claim.leaseToken) return@execute PhotoErasureOutcome.PENDING
            if (job.revision != claim.revision || photos.hasUnfinishedWork(claim.photoId, "ANY")) pending = true
            if (receipts.any { saved ->
                    intents.findById(saved.id).orElse(null)?.cleanupAfter?.toInstant()?.isAfter(clock.instant()) == true
                }) pending = true
            if (photo != null && photo.removalState != RemovalState.NONE && !job.retainTombstone) pending = true
            job.resourceOutcomes = outcomes
            job.faceOutcome = faceOutcome
            job.outcome = when { pending -> PhotoErasureOutcome.PENDING; failure != null -> PhotoErasureOutcome.RETRY_REQUIRED; else -> PhotoErasureOutcome.COMPLETE }
            job.lastError = failure
            job.leaseToken = null
            job.leaseUntil = null
            job.updatedAt = clock.instant()
            // Backoff is bounded; due ordering moves persistent failures behind other jobs.
            job.nextAttemptAt = clock.instant().plus(properties.cleanupBackoff.multipliedBy(minOf(job.attempts, 12).toLong()))
            if (job.outcome == PhotoErasureOutcome.COMPLETE) {
                if (photo != null) {
                    metadata.findById(photo.id).orElse(null)?.let(metadata::delete)
                    photo.facePersons.clear()
                    photo.bibs.clear()
                    photos.saveAndFlush(photo)
                    intents.scheduleCleanupForResultPhoto(photo.id, OffsetDateTime.now(clock))
                    if (job.retainTombstone) photos.minimizeErasedPhoto(photo.id, OffsetDateTime.now(clock))
                    else photos.delete(photo)
                }
                job.completedAt = clock.instant()
            }
            jobs.saveAndFlush(job)
            job.outcome
        }!!
    }

    private fun invalid(): Nothing = throw ConflictException(code = ErrorCodes.INVALID_STATE_TRANSITION,
        message = "Photo has no matching erasure authorization")

    private fun privacyAuthorization(command: PhotoErasureCommand) = ErasureAuthorization(command.authorizedActorId,
        command.requestId!!, command.itemId!!, command.reason.trim(), command.commandId, clock.instant())
}
