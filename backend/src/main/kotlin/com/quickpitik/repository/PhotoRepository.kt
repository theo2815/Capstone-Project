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
          AND (:bib = '' OR UPPER(b.bibNumber) = UPPER(:bib))
        ORDER BY p.capturedAt DESC NULLS LAST, p.uploadedAt DESC, p.id ASC
        """,
        countQuery = """
        SELECT COUNT(DISTINCT p) FROM Photo p
        LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND (:bib = '' OR UPPER(b.bibNumber) = UPPER(:bib))
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

    // Photographer-scoped library: every photo in the event uploaded by this
    // photographer, regardless of status (the photographer can manage HIDDEN
    // rows from their dashboard). Sort comes from the Pageable so the caller
    // can flip newest|oldest without a second query.
    @Query(
        """
        SELECT p FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
        """,
        countQuery = """
        SELECT COUNT(p) FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
        """,
    )
    fun findPhotographerLibrary(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        pageable: Pageable,
    ): Page<Photo>

    fun findFirstByIdAndPhotographerId(id: UUID, photographerId: UUID): Photo?
}
