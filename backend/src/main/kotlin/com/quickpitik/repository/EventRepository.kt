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

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT e FROM Event e WHERE e.id = :id AND e.createdBy = :createdBy AND e.deletedAt IS NULL")
    fun findOwnedByIdForUpdate(@Param("id") id: UUID, @Param("createdBy") createdBy: UUID): Event?

    // Admin review queue (V46): submissions + pending pricing changes, oldest first.
    fun findByReviewStatusInAndDeletedAtIsNullOrderByCreatedAtAsc(statuses: Collection<EventReviewStatus>): List<Event>

    fun countByReviewStatusInAndDeletedAtIsNull(statuses: Collection<EventReviewStatus>): Long

    // Row lock for approve/reject so two admins deciding the same event
    // serialize on the state check (same shape as DisputeRepository.findByIdForUpdate).
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT e FROM Event e WHERE e.id = :id")
    fun findByIdForReview(@Param("id") id: UUID): Event?

    // Unlocked scalar read: lets approve decide whether photo rows must be
    // locked before the event row, without loading the entity it locks next.
    @Query("SELECT e.reviewStatus FROM Event e WHERE e.id = :id")
    fun findReviewStatus(@Param("id") id: UUID): EventReviewStatus?

    // Public /events list: admin-created events only. Photographer-owned events
    // (createdBy != null) surface on the owner's profile / a direct link, never
    // the general list — so a foreign photographer never sees them in the upload
    // picker either (the picker's public catalog is this same query).
    // Sentinel values keep every named parameter typed (Postgres rejects nullable
    // unknown-typed parameters with "function lower(bytea) does not exist").
    @Query(
        """
        SELECT e FROM Event e
        WHERE e.deletedAt IS NULL
          AND e.createdBy IS NULL
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

    @Modifying
    @Query("UPDATE Event e SET e.photoCount = GREATEST(0, e.photoCount + :delta) WHERE e.id = :id")
    fun adjustPhotoCount(@Param("id") id: UUID, @Param("delta") delta: Int): Int
}
