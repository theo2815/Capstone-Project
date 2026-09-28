package com.quickpitik.service.photographer

import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.repository.PhotoMetadataRepository
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
    private val extractor: PhotoMetadataExtractor,
    private val transactions: TransactionTemplate,
    private val clock: Clock,
) {
    private val log = LoggerFactory.getLogger(javaClass)

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
