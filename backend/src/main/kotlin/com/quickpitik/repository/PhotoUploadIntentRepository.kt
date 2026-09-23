package com.quickpitik.repository

import com.quickpitik.entity.PhotoUploadIntent
import com.quickpitik.entity.PhotoUploadIntentState
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.time.OffsetDateTime
import java.util.UUID

@Repository
interface PhotoUploadIntentRepository : JpaRepository<PhotoUploadIntent, UUID> {

    @Query(
        "SELECT i FROM PhotoUploadIntent i " +
            "WHERE i.photoId = :photoId AND i.photographerId = :ownerId AND i.eventId = :eventId",
    )
    fun findOwned(
        @Param("photoId") photoId: UUID,
        @Param("ownerId") ownerId: UUID,
        @Param("eventId") eventId: UUID,
    ): PhotoUploadIntent?

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query(
        "SELECT i FROM PhotoUploadIntent i " +
            "WHERE i.photoId = :photoId AND i.photographerId = :ownerId AND i.eventId = :eventId",
    )
    fun findOwnedForUpdate(
        @Param("photoId") photoId: UUID,
        @Param("ownerId") ownerId: UUID,
        @Param("eventId") eventId: UUID,
    ): PhotoUploadIntent?

    fun findByPhotographerIdAndRequestId(photographerId: UUID, requestId: UUID): PhotoUploadIntent?

    @Query(
        value = """
            SELECT COUNT(*) AS slots, COALESCE(SUM(size_bytes), 0)::bigint AS bytes
            FROM photo_upload_intents
            WHERE photographer_id = :ownerId
              AND state IN ('PENDING', 'VERIFYING')
              AND expires_at > :now
        """,
        nativeQuery = true,
    )
    fun activeQuota(
        @Param("ownerId") ownerId: UUID,
        @Param("now") now: OffsetDateTime,
    ): PhotoUploadQuota

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET state = 'VERIFYING', lease_token = :leaseToken, lease_until = :leaseUntil, updated_at = :now
            WHERE photo_id = :photoId
              AND expires_at > :now
              AND (
                state = 'PENDING'
                OR (state = 'VERIFYING' AND (lease_until IS NULL OR lease_until <= :now))
              )
        """,
        nativeQuery = true,
    )
    fun acquireVerification(
        @Param("photoId") photoId: UUID,
        @Param("leaseToken") leaseToken: UUID,
        @Param("leaseUntil") leaseUntil: OffsetDateTime,
        @Param("now") now: OffsetDateTime,
    ): Int

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET state = 'COMMITTED', result_photo_id = :resultPhotoId, error_code = NULL,
                lease_token = NULL, lease_until = NULL, cleanup_after = :cleanupAfter, updated_at = :now
            WHERE photo_id = :photoId AND state = 'VERIFYING' AND lease_token = :leaseToken
        """,
        nativeQuery = true,
    )
    fun complete(
        @Param("photoId") photoId: UUID,
        @Param("leaseToken") leaseToken: UUID,
        @Param("resultPhotoId") resultPhotoId: UUID,
        @Param("cleanupAfter") cleanupAfter: OffsetDateTime,
        @Param("now") now: OffsetDateTime,
    ): Int

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET state = 'REJECTED', error_code = :errorCode,
                lease_token = NULL, lease_until = NULL, cleanup_after = :cleanupAfter, updated_at = :now
            WHERE photo_id = :photoId AND state = 'VERIFYING' AND lease_token = :leaseToken
        """,
        nativeQuery = true,
    )
    fun reject(
        @Param("photoId") photoId: UUID,
        @Param("leaseToken") leaseToken: UUID,
        @Param("errorCode") errorCode: String,
        @Param("cleanupAfter") cleanupAfter: OffsetDateTime,
        @Param("now") now: OffsetDateTime,
    ): Int

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET state = 'EXPIRED', lease_token = NULL, lease_until = NULL,
                cleanup_after = :cleanupAfter, updated_at = :now
            WHERE photo_id = :photoId
              AND expires_at <= :now
              AND (
                state = 'PENDING'
                OR (state = 'VERIFYING' AND (lease_until IS NULL OR lease_until <= :now))
              )
        """,
        nativeQuery = true,
    )
    fun expire(
        @Param("photoId") photoId: UUID,
        @Param("cleanupAfter") cleanupAfter: OffsetDateTime,
        @Param("now") now: OffsetDateTime,
    ): Int

    @Query(
        "SELECT i.photoId FROM PhotoUploadIntent i " +
            "WHERE i.expiresAt <= :now AND (" +
            "i.state = com.quickpitik.entity.PhotoUploadIntentState.PENDING OR (" +
            "i.state = com.quickpitik.entity.PhotoUploadIntentState.VERIFYING " +
            "AND (i.leaseUntil IS NULL OR i.leaseUntil <= :now))) " +
            "ORDER BY i.expiresAt, i.photoId",
    )
    fun findExpiredCandidateIds(
        @Param("now") now: OffsetDateTime,
        pageable: Pageable,
    ): List<UUID>

    @Query(
        "SELECT i.photoId FROM PhotoUploadIntent i " +
            "WHERE i.cleanupAfter IS NOT NULL AND i.cleanupAfter <= :now " +
            "AND i.state IN :states ORDER BY i.cleanupAfter, i.photoId",
    )
    fun findCleanupCandidateIds(
        @Param("now") now: OffsetDateTime,
        @Param("states") states: Collection<PhotoUploadIntentState>,
        pageable: Pageable,
    ): List<UUID>
}

interface PhotoUploadQuota {
    val slots: Long
    val bytes: Long
}
