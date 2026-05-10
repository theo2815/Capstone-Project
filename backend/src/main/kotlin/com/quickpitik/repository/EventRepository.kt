package com.quickpitik.repository

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
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

    fun countByStatusAndDeletedAtIsNull(status: EventStatus): Long

    // Admin event list — includes DRAFT (admin sees the full pipeline). State
    // filtering (live / upcoming / open / past) needs the date check, so it
    // happens service-side. status filter is service-side too since the FE
    // sends derived state, not raw status.
    @Query(
        """
        SELECT e FROM Event e
        WHERE e.deletedAt IS NULL
          AND (:search = '' OR
               LOWER(e.name) LIKE LOWER(CONCAT('%', :search, '%')) OR
               LOWER(e.location) LIKE LOWER(CONCAT('%', :search, '%')))
          AND e.date >= :dateFrom
          AND e.date <= :dateTo
        ORDER BY e.date DESC, e.id ASC
        """,
        countQuery = """
        SELECT COUNT(e) FROM Event e
        WHERE e.deletedAt IS NULL
          AND (:search = '' OR
               LOWER(e.name) LIKE LOWER(CONCAT('%', :search, '%')) OR
               LOWER(e.location) LIKE LOWER(CONCAT('%', :search, '%')))
          AND e.date >= :dateFrom
          AND e.date <= :dateTo
        """,
    )
    fun pageForAdmin(
        @Param("search") search: String,
        @Param("dateFrom") dateFrom: LocalDate,
        @Param("dateTo") dateTo: LocalDate,
        pageable: Pageable,
    ): Page<Event>

    // Atomic counter bump — concurrent uploads must NOT race on read-modify-write
    // of events.photo_count. The hero number on /events/[slug] under-counts when
    // two uploads read the same value and both write n+1 instead of n+2 (H-3).
    @Modifying
    @Query("UPDATE Event e SET e.photoCount = e.photoCount + 1 WHERE e.id = :id")
    fun incrementPhotoCount(@Param("id") id: UUID): Int
}
