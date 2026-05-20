package com.quickpitik.repository

import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface DownloadGrantRepository : JpaRepository<DownloadGrant, DownloadGrantId> {
    fun findByIdOrderId(orderId: UUID): List<DownloadGrant>

    // Subset of `photoIds` that the given user owns via an unexpired
    // DownloadGrant. Backs G-2 — the public event-photo list populates
    // PhotoDto.cleanUrl for these ids. Empty input returns empty (Hibernate
    // rejects `IN ()`).
    @Query(
        """
        SELECT dg.id.photoId FROM DownloadGrant dg
        WHERE dg.id.photoId IN :photoIds
          AND dg.grantedUntil > CURRENT_TIMESTAMP
          AND dg.id.orderId IN (
            SELECT o.id FROM Order o WHERE o.userId = :userId
          )
        """,
    )
    fun findOwnedPhotoIdsByUserAndPhotoIds(
        @Param("userId") userId: UUID,
        @Param("photoIds") photoIds: Collection<UUID>,
    ): List<UUID>
}
