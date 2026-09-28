package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photos.PhotoMetadataDto
import com.quickpitik.dto.photos.toDto
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.PhotoMetadataRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.image.PhotoMetadataExtractor
import com.quickpitik.service.image.hasRecordedTags
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID

@Service
class PhotoMetadataService(
    private val repository: PhotoMetadataRepository,
    private val photos: PhotoRepository,
    private val coverage: EventPhotographerRepository,
    private val extractor: PhotoMetadataExtractor,
    private val transactions: TransactionTemplate,
    private val clock: Clock,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(readOnly = true)
    fun getOwned(userId: UUID, photoId: UUID): PhotoMetadataDto {
        photos.findByIdAndPhotographerIdAndDeletionStartedAtIsNull(photoId, userId) ?: throw notFound()
        return repository.findById(photoId).orElse(null).toDto(photoId)
    }

    @Transactional(readOnly = true)
    fun getPublic(photoId: UUID): PhotoMetadataDto {
        val photo = photos.findById(photoId).orElse(null)
        if (photo == null || photo.status != PhotoStatus.LIVE || photo.deletedAt != null || photo.deletionStartedAt != null ||
            metadataAccess(photo.eventId, listOf(photo), null)[photo.id] != "public") {
            throw notFound()
        }
        return repository.findById(photoId).orElse(null).toDto(photoId)
    }

    @Transactional(readOnly = true)
    fun metadataAccess(eventId: UUID, photos: Collection<Photo>, requesterUserId: UUID?): Map<UUID, String> {
        val photographerIds = photos.mapNotNullTo(mutableSetOf()) { it.photographerId }
        val publicIds = if (photographerIds.isEmpty()) emptySet()
        else coverage.findPublicMetadataPhotographerIds(eventId, photographerIds).toSet()
        return photos.associate { photo ->
            photo.id to when {
                requesterUserId != null && photo.photographerId == requesterUserId -> "owner"
                photo.photographerId in publicIds -> "public"
                else -> "none"
            }
        }
    }

    private fun notFound() = NotFoundException(code = ErrorCodes.NOT_FOUND, message = "Photo not found")

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    fun extractIfDue(photoId: UUID, originalBytes: ByteArray) {
        try {
            val token = UUID.randomUUID()
            val now = OffsetDateTime.now(clock)
            val claimed = transactions.execute {
                repository.claimIfDue(photoId, token, EXTRACTOR_VERSION, now, now.plusMinutes(LEASE_MINUTES), MAX_ATTEMPTS)
            }
            if (claimed != 1) return

            // Only parsing failures are terminal. A database outage leaves the
            // committed lease for recovery instead of mislabelling the bytes.
            val fields = try {
                originalBytes.inputStream().use(extractor::extract)
            } catch (ex: Exception) {
                log.warn("Metadata parsing failed for photo {} ({})", photoId, ex.javaClass.simpleName)
                null
            }
            val state = when {
                fields == null -> PhotoMetadataState.FAILED
                fields.hasRecordedTags() -> PhotoMetadataState.READY
                else -> PhotoMetadataState.EMPTY
            }
            transactions.executeWithoutResult {
                repository.settle(photoId, token, EXTRACTOR_VERSION, state, fields, OffsetDateTime.now(clock))
            }
        } catch (ex: Exception) {
            log.warn("Metadata persistence failed for photo {} ({})", photoId, ex.javaClass.simpleName)
        }
    }

    private companion object {
        const val EXTRACTOR_VERSION = 1
        const val LEASE_MINUTES = 5L
        const val MAX_ATTEMPTS = 3
    }
}
