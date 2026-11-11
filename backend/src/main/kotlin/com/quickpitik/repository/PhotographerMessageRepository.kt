package com.quickpitik.repository

import com.quickpitik.entity.PhotographerMessage
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID

interface PhotographerMessageRepository : JpaRepository<PhotographerMessage, UUID> {

    fun findByNoticeKey(noticeKey: String): PhotographerMessage?

    fun findByPhotographerIdOrderByCreatedAtDescIdAsc(photographerId: UUID): List<PhotographerMessage>

    // List for /me/photographer/messages — filters soft-deleted rows. Paged:
    // the inbox used to return every row a photographer had ever received. The
    // derived ordering in the method name still applies; OffsetLimitPageable
    // passes Sort.unsorted() so it contributes offset/limit only.
    fun findByPhotographerIdAndRemovedAtIsNullOrderByCreatedAtDescIdAsc(
        photographerId: UUID,
        pageable: Pageable,
    ): List<PhotographerMessage>

    // Total un-removed messages — surfaced via the X-Total-Count header so the
    // web inbox knows the true count behind its capped page.
    fun countByPhotographerIdAndRemovedAtIsNull(photographerId: UUID): Long

    // Tenant-safe single-row fetch — guards against IDOR by matching photographer_id.
    fun findByIdAndPhotographerId(
        id: UUID,
        photographerId: UUID,
    ): Optional<PhotographerMessage>

    @Modifying
    @Query(
        "UPDATE PhotographerMessage m SET m.readAt = :readAt " +
            "WHERE m.photographerId = :photographerId AND m.readAt IS NULL AND m.removedAt IS NULL",
    )
    fun markAllReadForPhotographer(
        @Param("photographerId") photographerId: UUID,
        @Param("readAt") readAt: OffsetDateTime,
    ): Int

    // RD-10 email send claim (columns from V99, the orders.email_sent_at pattern). Native: the entity does not map them.
    @Modifying
    @Query(value = "UPDATE photographer_messages SET email_sent_at = :now WHERE id = :id AND email_sent_at IS NULL", nativeQuery = true)
    fun claimEmailSend(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    @Modifying
    @Query(value = "UPDATE photographer_messages SET email_delivered_at = :at WHERE id = :id AND email_delivered_at IS NULL", nativeQuery = true)
    fun stampEmailDelivered(@Param("id") id: UUID, @Param("at") at: OffsetDateTime): Int

    // Removal notices claimed before the cutoff and never delivered, from the last 24 hours.
    @Query(
        value = """SELECT id FROM photographer_messages
            WHERE kind IN ('photo_removal_request', 'photo_removal_outcome')
              AND email_sent_at < :cutoff AND email_delivered_at IS NULL AND created_at >= :since
            ORDER BY email_sent_at LIMIT :limit""",
        nativeQuery = true,
    )
    fun findStaleEmailClaimIds(@Param("cutoff") cutoff: OffsetDateTime, @Param("since") since: OffsetDateTime, @Param("limit") limit: Int): List<UUID>

    // Match the old timestamp and missing delivery stamp, so a second instance's fresh claim is not released.
    @Modifying
    @Query(
        value = "UPDATE photographer_messages SET email_sent_at = NULL WHERE id = :id AND email_sent_at < :cutoff AND email_delivered_at IS NULL",
        nativeQuery = true,
    )
    fun releaseStaleEmailSend(@Param("id") id: UUID, @Param("cutoff") cutoff: OffsetDateTime): Int
}
