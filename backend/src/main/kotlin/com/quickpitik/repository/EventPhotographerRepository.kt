package com.quickpitik.repository

import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface EventPhotographerRepository : JpaRepository<EventPhotographer, EventPhotographerId> {

    @Query(
        """
        SELECT ep FROM EventPhotographer ep
        WHERE ep.id.photographerId = :photographerId
          AND (:withUploadsOnly = false OR ep.photoCount > 0)
        ORDER BY ep.lastUploadAt DESC NULLS LAST, ep.joinedAt DESC, ep.id.eventId ASC
        """,
        countQuery = """
        SELECT COUNT(ep) FROM EventPhotographer ep
        WHERE ep.id.photographerId = :photographerId
          AND (:withUploadsOnly = false OR ep.photoCount > 0)
        """,
    )
    fun searchForPhotographer(
        @Param("photographerId") photographerId: UUID,
        @Param("withUploadsOnly") withUploadsOnly: Boolean,
        pageable: Pageable,
    ): Page<EventPhotographer>

    fun findAllByIdPhotographerId(photographerId: UUID): List<EventPhotographer>
}
