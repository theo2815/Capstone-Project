package com.quickpitik.repository

import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface PhotoRepository : JpaRepository<Photo, UUID> {

    @Query(
        """
        SELECT DISTINCT p FROM Photo p
        LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND (:bib = '' OR UPPER(b.bibNumber) LIKE CONCAT(UPPER(:bib), '%'))
        ORDER BY p.capturedAt DESC NULLS LAST, p.uploadedAt DESC, p.id ASC
        """,
        countQuery = """
        SELECT COUNT(DISTINCT p) FROM Photo p
        LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND (:bib = '' OR UPPER(b.bibNumber) LIKE CONCAT(UPPER(:bib), '%'))
        """,
    )
    fun searchForEvent(
        @Param("eventId") eventId: UUID,
        @Param("status") status: PhotoStatus,
        @Param("bib") bib: String,
        pageable: Pageable,
    ): Page<Photo>

    @Query(
        """
        SELECT DISTINCT p FROM Photo p
        JOIN p.facePersons fp
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND fp.aiPersonId IN :aiPersonIds
        ORDER BY p.capturedAt DESC NULLS LAST, p.uploadedAt DESC, p.id ASC
        """,
        countQuery = """
        SELECT COUNT(DISTINCT p) FROM Photo p
        JOIN p.facePersons fp
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND fp.aiPersonId IN :aiPersonIds
        """,
    )
    fun findByEventAndPersonIds(
        @Param("eventId") eventId: UUID,
        @Param("status") status: PhotoStatus,
        @Param("aiPersonIds") aiPersonIds: Collection<String>,
        pageable: Pageable,
    ): Page<Photo>
}
