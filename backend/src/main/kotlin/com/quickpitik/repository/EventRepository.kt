package com.quickpitik.repository

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.LocalDate
import java.util.UUID

interface EventRepository : JpaRepository<Event, UUID> {
    fun findBySlugAndDeletedAtIsNull(slug: String): Event?

    // Public read of one event (V46): a DRAFT — a pending or rejected
    // photographer submission — is not a public page. An UNLISTED live event
    // still is: that is what "link-only" means.
    @Query(
        """
        SELECT e FROM Event e
        WHERE e.slug = :slug
          AND e.deletedAt IS NULL
          AND e.status <> com.quickpitik.entity.EventStatus.DRAFT
        """,
    )
    fun findPublicBySlug(@Param("slug") slug: String): Event?

    // Owner-scoped lookup for photographer-owned events (V46): the tenant
    // filter lives in the query, so a foreign event id simply misses → 404.
    fun findByIdAndCreatedByAndDeletedAtIsNull(id: UUID, createdBy: UUID): Event?

    // Admin review queue (V46): submissions + pending pricing changes, oldest first.
    fun findByReviewStatusInAndDeletedAtIsNullOrderByCreatedAtAsc(statuses: Collection<EventReviewStatus>): List<Event>

    fun countByReviewStatusInAndDeletedAtIsNull(statuses: Collection<EventReviewStatus>): Long

    // Row lock for approve/reject so two admins deciding the same event
    // serialize on the state check (same shape as DisputeRepository.findByIdForUpdate).
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT e FROM Event e WHERE e.id = :id")
    fun findByIdForReview(@Param("id") id: UUID): Event?

    // Sentinel values keep every named parameter typed (Postgres rejects nullable
    // unknown-typed parameters with "function lower(bytea) does not exist").
    @Query(
        """
        SELECT e FROM Event e
        WHERE e.deletedAt IS NULL
          AND e.status IN :statuses
          AND e.visibility = com.quickpitik.entity.EventVisibility.PUBLIC
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
