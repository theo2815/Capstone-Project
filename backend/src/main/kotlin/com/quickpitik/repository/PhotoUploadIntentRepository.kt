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
import java.time.Instant
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

    // Native clients send no request id and begin again after a failed PUT.
    // Give them the same reservation while its signed PUT still has time left,
    // instead of reserving a new one on every retry.
    @Query(
        value = """
            SELECT * FROM photo_upload_intents
            WHERE photographer_id = :ownerId AND event_id = :eventId
              AND content_hash = :contentHash AND content_type = :contentType
              AND size_bytes = :sizeBytes AND request_id IS NULL
              AND state = 'PENDING' AND put_expires_at > :minPutExpiry AND expires_at > :now
            ORDER BY put_expires_at DESC, photo_id
            LIMIT 1
        """,
        nativeQuery = true,
    )
    fun findReusablePending(
        @Param("ownerId") ownerId: UUID,
        @Param("eventId") eventId: UUID,
        @Param("contentHash") contentHash: String,
        @Param("contentType") contentType: String,
        @Param("sizeBytes") sizeBytes: Long,
        @Param("minPutExpiry") minPutExpiry: OffsetDateTime,
        @Param("now") now: OffsetDateTime,
    ): PhotoUploadIntent?

    // Admission counts only reservations that can still take bytes or are being
    // verified now. An abandoned PENDING row stays committable for its 24 h intent
    // TTL but must not hold one of the owner's slots after its signed PUT expires.
    @Query(
        value = """
            SELECT COUNT(*) AS slots, COALESCE(SUM(size_bytes), 0)::bigint AS bytes
            FROM photo_upload_intents
            WHERE photographer_id = :ownerId
              AND state IN ('PENDING', 'VERIFYING')
              AND expires_at > :now
              AND (put_expires_at > :now OR (state = 'VERIFYING' AND lease_until > :now))
        """,
        nativeQuery = true,
    )
    fun activeQuota(
        @Param("ownerId") ownerId: UUID,
        @Param("now") now: OffsetDateTime,
    ): PhotoUploadQuota

    @Query(
        value = """
            SELECT COUNT(*) AS slots,
                   COALESCE(SUM(size_bytes), 0)::bigint AS bytes,
                   MIN(created_at) AS "oldestCreatedAt"
            FROM photo_upload_intents
            WHERE state IN ('PENDING', 'VERIFYING') AND expires_at > :now
        """,
        nativeQuery = true,
    )
    fun pendingMetrics(@Param("now") now: OffsetDateTime): PhotoUploadPendingMetrics?

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
            SET state = 'PENDING', lease_token = NULL, lease_until = NULL, updated_at = :now
            WHERE photo_id = :photoId AND state = 'VERIFYING' AND lease_token = :leaseToken
        """,
        nativeQuery = true,
    )
    fun releaseVerification(
        @Param("photoId") photoId: UUID,
        @Param("leaseToken") leaseToken: UUID,
        @Param("now") now: OffsetDateTime,
    ): Int

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET state = 'EXPIRED', lease_token = NULL, lease_until = NULL,
                cleanup_after = :cleanupAfter, updated_at = :now
            WHERE photo_id = :photoId AND state = 'VERIFYING' AND lease_token = :leaseToken
              AND put_expires_at <= :now
        """,
        nativeQuery = true,
    )
    fun expireMissingAfterPut(
        @Param("photoId") photoId: UUID,
        @Param("leaseToken") leaseToken: UUID,
        @Param("cleanupAfter") cleanupAfter: OffsetDateTime,
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

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET cleanup_after = :claimedUntil, cleanup_checked_at = :now, updated_at = :now
            WHERE photo_id = :photoId
              AND cleanup_after IS NOT NULL AND cleanup_after <= :now
              AND state IN ('COMMITTED', 'REJECTED', 'EXPIRED')
        """,
        nativeQuery = true,
    )
    fun claimCleanup(
        @Param("photoId") photoId: UUID,
        @Param("now") now: OffsetDateTime,
        @Param("claimedUntil") claimedUntil: OffsetDateTime,
    ): Int

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET cleaned_at = :cleanedAt, cleanup_after = :nextCleanupAfter,
                cleanup_checked_at = :now, updated_at = :now
            WHERE photo_id = :photoId AND cleanup_after = :claimedUntil AND cleanup_checked_at = :now
              AND state IN ('COMMITTED', 'REJECTED', 'EXPIRED')
        """,
        nativeQuery = true,
    )
    fun completeCleanup(
        @Param("photoId") photoId: UUID,
        @Param("claimedUntil") claimedUntil: OffsetDateTime,
        @Param("cleanedAt") cleanedAt: OffsetDateTime,
        @Param("nextCleanupAfter") nextCleanupAfter: OffsetDateTime?,
        @Param("now") now: OffsetDateTime,
    ): Int

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET cleanup_after = :retryAt, cleanup_checked_at = :now, updated_at = :now
            WHERE photo_id = :photoId AND cleanup_after = :claimedUntil AND cleanup_checked_at = :now
              AND state IN ('COMMITTED', 'REJECTED', 'EXPIRED')
        """,
        nativeQuery = true,
    )
    fun retryCleanup(
        @Param("photoId") photoId: UUID,
        @Param("claimedUntil") claimedUntil: OffsetDateTime,
        @Param("retryAt") retryAt: OffsetDateTime,
        @Param("now") now: OffsetDateTime,
    ): Int

    @Modifying(clearAutomatically = true, flushAutomatically = true)
    @Query(
        value = """
            UPDATE photo_upload_intents
            SET cleanup_after = :now, cleanup_checked_at = NULL, cleaned_at = NULL, updated_at = :now
            WHERE result_photo_id = :photoId AND state = 'COMMITTED'
        """,
        nativeQuery = true,
    )
    fun scheduleCleanupForResultPhoto(
        @Param("photoId") photoId: UUID,
        @Param("now") now: OffsetDateTime,
    ): Int
}

interface PhotoUploadQuota {
    val slots: Long
    val bytes: Long
}

interface PhotoUploadPendingMetrics {
    val slots: Long
    val bytes: Long
    val oldestCreatedAt: Instant?
}
