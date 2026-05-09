package com.quickpitik.repository

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.LocalDate
import java.util.UUID

interface EventRepository : JpaRepository<Event, UUID> {
    fun findBySlugAndDeletedAtIsNull(slug: String): Event?

    // Sentinel values keep every named parameter typed (Postgres rejects nullable
    // unknown-typed parameters with "function lower(bytea) does not exist").
    @Query(
        """
        SELECT e FROM Event e
        WHERE e.deletedAt IS NULL
          AND e.status IN :statuses
          AND (:search = '' OR
               LOWER(e.name) LIKE LOWER(CONCAT('%', :search, '%')) OR
               LOWER(e.location) LIKE LOWER(CONCAT('%', :search, '%')))
          AND (:city = '' OR LOWER(e.location) LIKE LOWER(CONCAT('%, ', :city)))
          AND e.date >= :dateFrom
          AND e.date <= :dateTo
        ORDER BY e.date DESC, e.id ASC
        """,
    )
    fun search(
        @Param("statuses") statuses: Collection<EventStatus>,
        @Param("search") search: String,
        @Param("city") city: String,
        @Param("dateFrom") dateFrom: LocalDate,
        @Param("dateTo") dateTo: LocalDate,
        pageable: Pageable,
    ): Page<Event>
}
