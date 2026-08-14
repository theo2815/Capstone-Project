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

    fun findByPhotographerIdOrderByCreatedAtDescIdAsc(photographerId: UUID): List<PhotographerMessage>

    // List for /me/photographer/messages — filters soft-deleted rows. Paged:
    // the inbox used to return every row a photographer had ever received. The
    // derived ordering in the method name still applies; OffsetLimitPageable
    // passes Sort.unsorted() so it contributes offset/limit only.
    fun findByPhotographerIdAndRemovedAtIsNullOrderByCreatedAtDescIdAsc(
        photographerId: UUID,
        pageable: Pageable,
    ): List<PhotographerMessage>

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
}
