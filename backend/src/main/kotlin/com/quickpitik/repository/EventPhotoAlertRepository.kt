package com.quickpitik.repository

import com.quickpitik.entity.EventPhotoAlert
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

@Repository
interface EventPhotoAlertRepository : JpaRepository<EventPhotoAlert, UUID> {

    fun findByEventIdAndUserId(eventId: UUID, userId: UUID): EventPhotoAlert?

    @Modifying
    fun deleteByEventIdAndUserId(eventId: UUID, userId: UUID): Long

    // Un-notified opt-ins with no active claim. Ordinary rows stay inside the
    // event window; expired claims remain recoverable after it. Never-checked
    // rows sort first, then least recently checked, so bounded batches rotate.
    @Query(
        """
        SELECT a FROM EventPhotoAlert a
        WHERE a.notifiedAt IS NULL
          AND (a.notificationClaimedAt IS NULL OR a.notificationClaimedAt <= :claimExpiredBefore)
          AND EXISTS (
              SELECT e.id FROM Event e
              WHERE e.id = a.eventId
                AND e.deletedAt IS NULL
                AND e.status IN (
                    com.quickpitik.entity.EventStatus.ACTIVE,
                    com.quickpitik.entity.EventStatus.COMPLETED
                )
                AND (
                    a.notificationClaimedAt IS NOT NULL
                    OR (e.date <= :today AND e.date >= :windowStart)
                )
          )
        ORDER BY
          CASE WHEN a.lastCheckedAt IS NULL THEN 0 ELSE 1 END,
          a.lastCheckedAt ASC,
          a.createdAt ASC,
          a.id ASC
        """,
    )
    fun findPendingInWindow(
        @Param("today") today: LocalDate,
        @Param("windowStart") windowStart: LocalDate,
        @Param("claimExpiredBefore") claimExpiredBefore: OffsetDateTime,
        pageable: Pageable,
    ): List<EventPhotoAlert>

    // The claim timestamp is also the attempt token used by the delivery stamp.
    @Modifying
    @Query(
        "UPDATE EventPhotoAlert a SET a.notificationClaimedAt = :claimToken " +
            "WHERE a.id = :id AND a.notifiedAt IS NULL " +
            "AND (a.notificationClaimedAt IS NULL OR a.notificationClaimedAt <= :claimExpiredBefore)",
    )
    fun claimNotification(
        @Param("id") id: UUID,
        @Param("claimToken") claimToken: OffsetDateTime,
        @Param("claimExpiredBefore") claimExpiredBefore: OffsetDateTime,
    ): Int

    // A stale worker cannot stamp delivery or clear a replacement worker's claim.
    @Modifying
    @Query(
        "UPDATE EventPhotoAlert a SET a.notifiedAt = :deliveredAt, a.notificationClaimedAt = NULL " +
            "WHERE a.id = :id AND a.notifiedAt IS NULL AND a.notificationClaimedAt = :claimToken",
    )
    fun stampNotificationDelivered(
        @Param("id") id: UUID,
        @Param("claimToken") claimToken: OffsetDateTime,
        @Param("deliveredAt") deliveredAt: OffsetDateTime,
    ): Int
}
