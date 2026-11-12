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
import java.time.OffsetDateTime
import java.util.UUID

interface EventRepository : JpaRepository<Event, UUID> {
    fun findBySlugAndDeletedAtIsNull(slug: String): Event?

    // EventFaceExpiry: events whose face templates are still held and are due
    // for deletion (dated on or before the cutoff, or deleted).
    fun findByFacesDeletedAtIsNullAndDateLessThanEqual(cutoff: LocalDate): List<Event>
    fun findByFacesDeletedAtIsNullAndDeletedAtIsNotNull(): List<Event>

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

    // Public events list. By default: admin-created events only — the upload
    // pickers (website, mobile, desktop) use this same query and rely on that,
    // so a foreign photographer never sees another's event there.
    // includeListed=true (sent only by the /events catalog and mobile runner
    // discovery) also returns photographer events an admin listed (V89), unless
    // that photographer is suspended (AA-I12a, = PhotographerSuspension).
    // A hidden shared race (V96 hidden_at, AA-R1) leaves both; its slug still resolves.
    // Sentinel values keep every named parameter typed (Postgres rejects nullable
    // unknown-typed parameters with "function lower(bytea) does not exist").
    @Query(
        """
        SELECT e FROM Event e
        WHERE e.deletedAt IS NULL
          AND e.hiddenAt IS NULL
          AND (e.createdBy IS NULL OR (:includeListed = TRUE AND e.listedOnEvents = TRUE
               AND NOT EXISTS (SELECT su.id FROM User su WHERE su.id = e.createdBy AND su.suspendedAt IS NOT NULL)))
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
        @Param("includeListed") includeListed: Boolean,
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

    // AA-R8 race proof sweep: typed races whose upload window closed and that were never checked.
    @Query(
        """
        SELECT e FROM Event e
        WHERE e.eventType IS NOT NULL AND e.deletedAt IS NULL
          AND e.status <> com.quickpitik.entity.EventStatus.DRAFT
          AND e.proofCheckedAt IS NULL AND e.date <= :closedOnOrBefore
        ORDER BY e.date, e.id
        """,
    )
    fun findRacesDueForProof(@Param("closedOnOrBefore") closedOnOrBefore: LocalDate, pageable: Pageable): List<Event>

    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query("UPDATE Event e SET e.proofCheckedAt = :now WHERE e.id = :id AND e.proofCheckedAt IS NULL")
    fun claimProofCheck(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    // An admin approval or relist (admin_approved_at) always wins over a hide.
    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query(
        """
        UPDATE Event e SET e.hiddenAt = :now
        WHERE e.id = :id AND e.eventType IS NOT NULL AND e.hiddenAt IS NULL AND e.adminApprovedAt IS NULL
        """,
    )
    fun hideUnprovenRace(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    // AA-R3: every race create (and R5's edit) runs "lock, re-check, insert" under this one
    // transaction-scoped lock, so two simultaneous creates cannot both miss each other.
    @Query(value = "SELECT CAST(pg_advisory_xact_lock($RACE_CREATION_LOCK_KEY) AS text)", nativeQuery = true)
    fun lockRaceCreation(): String?

    // Duplicate candidates: live shared events in a date range, legacy admin events included.
    @Query(
        """
        SELECT e FROM Event e
        WHERE e.date BETWEEN :from AND :to AND e.deletedAt IS NULL AND e.createdBy IS NULL
          AND e.reviewStatus <> com.quickpitik.entity.EventReviewStatus.REJECTED
        """,
    )
    fun findRaceCandidates(@Param("from") from: LocalDate, @Param("to") to: LocalDate): List<Event>

    @Query(
        value = "SELECT event_id AS eventId, name_key AS nameKey FROM event_name_aliases WHERE event_id IN (:eventIds)",
        nativeQuery = true,
    )
    fun findAliasKeys(@Param("eventIds") eventIds: Collection<UUID>): List<AliasKeyRow>

    @Modifying
    @Query(
        value = """
        INSERT INTO event_name_aliases (event_id, name_key, added_by) VALUES (:eventId, :nameKey, :addedBy)
        ON CONFLICT (event_id, name_key) DO NOTHING
        """,
        nativeQuery = true,
    )
    fun insertAlias(@Param("eventId") eventId: UUID, @Param("nameKey") nameKey: String, @Param("addedBy") addedBy: UUID): Int

    // AA-R5: the submitter's race, locked for an edit. Rejected and deleted races are not editable.
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query(
        "SELECT e FROM Event e WHERE e.id = :id AND e.submittedBy = :submittedBy AND e.eventType IS NOT NULL " +
            "AND e.deletedAt IS NULL AND e.reviewStatus <> com.quickpitik.entity.EventReviewStatus.REJECTED",
    )
    fun findRaceForSubmitterForUpdate(@Param("id") id: UUID, @Param("submittedBy") submittedBy: UUID): Event?

    // AA-R4's daily create cap reads this.
    @Query("SELECT COUNT(e) FROM Event e WHERE e.submittedBy = :submittedBy AND e.createdAt >= :since")
    fun countRacesSubmittedSince(@Param("submittedBy") submittedBy: UUID, @Param("since") since: OffsetDateTime): Long

    // AA-R11: the race price read fresh after the price locks, never from an entity loaded earlier.
    @Query("SELECT e.pricePerPhoto FROM Event e WHERE e.id = :id")
    fun findPricePerPhoto(@Param("id") id: UUID): java.math.BigDecimal?
}

// Not 78202: AdminTransitionConcurrencyIntegrationTest uses that key.
const val RACE_CREATION_LOCK_KEY = 78301L

interface AliasKeyRow {
    fun getEventId(): UUID
    fun getNameKey(): String
}
