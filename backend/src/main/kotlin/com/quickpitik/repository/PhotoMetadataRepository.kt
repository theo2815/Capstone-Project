package com.quickpitik.repository

import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.PhotoMetadata
import com.quickpitik.entity.PhotoMetadataState
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import java.time.OffsetDateTime
import java.util.UUID

interface PhotoMetadataRepository : JpaRepository<PhotoMetadata, UUID> {
    @Query("""
        SELECT m.photoId FROM PhotoMetadata m, Photo p
        WHERE p.id = m.photoId AND p.deletionStartedAt IS NULL
            AND m.extractorVersion = :version
            AND ((m.attempts < :maxAttempts
                    AND ((m.state = com.quickpitik.entity.PhotoMetadataState.PENDING
                            AND p.uploadedAt <= :pendingBefore
                            AND (m.nextAttemptAt IS NULL OR m.nextAttemptAt <= :now))
                        OR (m.state = com.quickpitik.entity.PhotoMetadataState.FAILED
                            AND m.nextAttemptAt IS NOT NULL AND m.nextAttemptAt <= :now)
                        OR (m.state = com.quickpitik.entity.PhotoMetadataState.PROCESSING
                            AND m.leaseUntil <= :now)))
                OR (m.state = com.quickpitik.entity.PhotoMetadataState.PROCESSING
                    AND m.attempts >= :maxAttempts AND m.leaseUntil <= :now))
        ORDER BY p.uploadedAt, m.photoId
    """)
    fun findDueCandidateIds(
        version: Int, maxAttempts: Int, pendingBefore: OffsetDateTime, now: OffsetDateTime, pageable: Pageable,
    ): List<UUID>

    @Modifying
    @Query("""
        UPDATE PhotoMetadata m SET m.state = com.quickpitik.entity.PhotoMetadataState.PROCESSING,
            m.leaseToken = :token, m.leaseUntil = :leaseUntil, m.attempts = m.attempts + 1,
            m.nextAttemptAt = null
        WHERE m.photoId = :photoId AND m.extractorVersion = :version AND m.attempts < :maxAttempts
            AND (m.nextAttemptAt IS NULL OR m.nextAttemptAt <= :now)
            AND (m.state = com.quickpitik.entity.PhotoMetadataState.PENDING
                OR (m.state = com.quickpitik.entity.PhotoMetadataState.PROCESSING AND m.leaseUntil <= :now)
                OR (m.state = com.quickpitik.entity.PhotoMetadataState.FAILED AND m.nextAttemptAt IS NOT NULL))
            AND EXISTS (SELECT p.id FROM Photo p WHERE p.id = m.photoId AND p.deletionStartedAt IS NULL)
    """)
    fun claimIfDue(photoId: UUID, token: UUID, version: Int, now: OffsetDateTime, leaseUntil: OffsetDateTime, maxAttempts: Int): Int

    @Modifying
    @Query("""
        UPDATE PhotoMetadata m SET m.state = :state, m.fields = :fields, m.extractedAt = :now,
            m.originalSizeBytes = COALESCE(m.originalSizeBytes, :sizeBytes),
            m.leaseToken = null, m.leaseUntil = null, m.nextAttemptAt = null
        WHERE m.photoId = :photoId AND m.leaseToken = :token AND m.extractorVersion = :version
            AND m.state = com.quickpitik.entity.PhotoMetadataState.PROCESSING
            AND EXISTS (SELECT p.id FROM Photo p WHERE p.id = m.photoId AND p.deletionStartedAt IS NULL)
    """)
    fun settle(photoId: UUID, token: UUID, version: Int, state: PhotoMetadataState, fields: PhotoMetadataFields?,
               now: OffsetDateTime, sizeBytes: Long? = null): Int

    @Modifying
    @Query("""
        UPDATE PhotoMetadata m SET m.state = com.quickpitik.entity.PhotoMetadataState.FAILED,
            m.extractedAt = :now, m.leaseToken = null, m.leaseUntil = null, m.nextAttemptAt = null
        WHERE m.photoId = :photoId AND m.extractorVersion = :version
            AND m.state = com.quickpitik.entity.PhotoMetadataState.PROCESSING
            AND m.attempts >= :maxAttempts AND m.leaseUntil <= :now
            AND EXISTS (SELECT p.id FROM Photo p WHERE p.id = m.photoId AND p.deletionStartedAt IS NULL)
    """)
    fun failExpiredExhaustedLease(photoId: UUID, version: Int, maxAttempts: Int, now: OffsetDateTime): Int

    @Modifying
    @Query("""
        UPDATE PhotoMetadata m SET m.state = com.quickpitik.entity.PhotoMetadataState.FAILED,
            m.leaseToken = null, m.leaseUntil = null, m.extractedAt = :now,
            m.nextAttemptAt = CASE WHEN m.attempts = 1 THEN :first
                WHEN m.attempts = 2 THEN :second WHEN m.attempts = 3 THEN :third ELSE m.nextAttemptAt END
        WHERE m.photoId = :photoId AND m.leaseToken = :token AND m.extractorVersion = :version
            AND m.state = com.quickpitik.entity.PhotoMetadataState.PROCESSING
            AND EXISTS (SELECT p.id FROM Photo p WHERE p.id = m.photoId AND p.deletionStartedAt IS NULL)
    """)
    fun settleStorageFailure(
        photoId: UUID, token: UUID, version: Int, now: OffsetDateTime,
        first: OffsetDateTime, second: OffsetDateTime, third: OffsetDateTime,
    ): Int

    @Modifying
    @Query("""
        UPDATE PhotoMetadata m SET m.state = com.quickpitik.entity.PhotoMetadataState.UNAVAILABLE,
            m.originalSizeBytes = COALESCE(:sizeBytes, m.originalSizeBytes), m.extractedAt = :now,
            m.leaseToken = null, m.leaseUntil = null, m.nextAttemptAt = null
        WHERE m.photoId = :photoId AND m.leaseToken = :token AND m.extractorVersion = :version
            AND m.state = com.quickpitik.entity.PhotoMetadataState.PROCESSING
            AND EXISTS (SELECT p.id FROM Photo p WHERE p.id = m.photoId AND p.deletionStartedAt IS NULL)
    """)
    fun settleUnavailable(photoId: UUID, token: UUID, version: Int, sizeBytes: Long?, now: OffsetDateTime): Int
}
