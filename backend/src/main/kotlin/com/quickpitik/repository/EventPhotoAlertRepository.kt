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

    // Un-notified opt-ins whose event is ACTIVE and inside its [date, date+3d]
    // upload window (windowStart = today - 3). The window bound is what stops the
    // sweep from re-running ai-api forever on a runner who was never
    // photographed — once the event's date leaves the window the row drops out.
    @Query(
        """
        SELECT a FROM EventPhotoAlert a
        WHERE a.notifiedAt IS NULL
          AND a.eventId IN (
              SELECT e.id FROM Event e
              WHERE e.deletedAt IS NULL
                AND e.status = com.quickpitik.entity.EventStatus.ACTIVE
                AND e.date <= :today
                AND e.date >= :windowStart
          )
        ORDER BY a.createdAt ASC
        """,
    )
    fun findPendingInWindow(
        @Param("today") today: LocalDate,
        @Param("windowStart") windowStart: LocalDate,
        pageable: Pageable,
    ): List<EventPhotoAlert>

    // Atomically claim the right to send this alert's email — mirrors
    // OrderRepository.claimReceiptSend. Returns 1 if this caller won the claim,
    // 0 if someone else already holds it. The WHERE clause makes the check and
    // the write one statement, so two concurrent sweeps can't both send.
    @Modifying
    @Query("UPDATE EventPhotoAlert a SET a.notifiedAt = :now WHERE a.id = :id AND a.notifiedAt IS NULL")
    fun claimNotify(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    // Give the claim back when the send itself failed, so the next sweep retries.
    @Modifying
    @Query("UPDATE EventPhotoAlert a SET a.notifiedAt = NULL WHERE a.id = :id")
    fun releaseNotify(@Param("id") id: UUID): Int
}
