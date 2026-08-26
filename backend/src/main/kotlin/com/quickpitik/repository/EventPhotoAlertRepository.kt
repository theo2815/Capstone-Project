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

    // Un-notified opt-ins for uploadable events, including one post-window day
    // so the last accepted uploads can finish indexing. Never-checked rows sort
    // first, then the least recently checked, so a batch cannot starve later
    // runners when early opt-ins have no match.
    @Query(
        """
        SELECT a FROM EventPhotoAlert a
        WHERE a.notifiedAt IS NULL
          AND a.eventId IN (
              SELECT e.id FROM Event e
              WHERE e.deletedAt IS NULL
                AND e.status IN (
                    com.quickpitik.entity.EventStatus.ACTIVE,
                    com.quickpitik.entity.EventStatus.COMPLETED
                )
                AND e.date <= :today
                AND e.date >= :windowStart
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
