package com.quickpitik.repository

import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

interface PhotoRepository : JpaRepository<Photo, UUID> {

    // Bulk re-prices every photo under the event. Used by admin event PATCH
    // when the operator changes events.price_per_photo so existing photos
    // pick up the new price across runner-facing galleries. Cart drift is
    // handled separately by CartService.add → CART_ITEM_PRICE_CHANGED at
    // checkout — we deliberately do not touch cart_items.price_php_at_add
    // so the server-canonical snapshot guarantee stays intact.
    @Modifying
    @Query("UPDATE Photo p SET p.pricePhp = :price WHERE p.eventId = :eventId")
    fun updatePriceByEventId(
        @Param("eventId") eventId: UUID,
        @Param("price") price: BigDecimal,
    ): Int

    // Bib filter is a substring (contains) match — OCR routinely clips digits
    // (e.g. "7202" stored as "720" or "71830" when the bib is "183") so exact
    // match is too brittle for real-world race photos. Substring also lets a
    // runner type just "183" if they only remember part of their bib. Anchored
    // prefix would be the next step up if false-positives become an issue.
    @Query(
        """
        SELECT DISTINCT p FROM Photo p
        LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND (:bib = '' OR UPPER(b.bibNumber) LIKE CONCAT('%', UPPER(:bib), '%'))
        ORDER BY p.capturedAt DESC NULLS LAST, p.uploadedAt DESC, p.id ASC
        """,
        countQuery = """
        SELECT COUNT(DISTINCT p) FROM Photo p
        LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND (:bib = '' OR UPPER(b.bibNumber) LIKE CONCAT('%', UPPER(:bib), '%'))
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

    // Status-filtered variant for the public gallery, where HIDDEN / PROCESSING
    // rows must not be returned AND must not inflate `total`. Applying the
    // status predicate at the query layer keeps the count accurate so the FE's
    // pagination doesn't jump to phantom pages.
    @Query(
        """
        SELECT p FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
          AND p.status = :status
        """,
        countQuery = """
        SELECT COUNT(p) FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
          AND p.status = :status
        """,
    )
    fun findPhotographerLibraryByStatus(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("status") status: PhotoStatus,
        pageable: Pageable,
    ): Page<Photo>

    fun findFirstByIdAndPhotographerId(id: UUID, photographerId: UUID): Photo?

    // Duplicate detection: the photographer's existing photo with this exact
    // content hash, if any. Backed by the partial unique index
    // uq_photos_photographer_content_hash (migration V24), so at most one row
    // matches. PhotoUploadService uses it to make uploads idempotent on content
    // — a same-event re-upload returns the existing photo, a different-event hit
    // is rejected.
    fun findFirstByPhotographerIdAndContentHash(photographerId: UUID, contentHash: String): Photo?

    // Batch pre-flight (dedup Phase 2): every photo of this photographer whose
    // content hash is in the given set, across ALL their events. The client
    // hashes files locally and calls this before sending bytes — a hash already
    // present is skipped (same event) or flagged (different event) without a
    // wasted upload. NULL hashes never match an IN list, so legacy rows are
    // naturally excluded. Backed by uq_photos_photographer_content_hash (V24).
    fun findByPhotographerIdAndContentHashIn(
        photographerId: UUID,
        contentHashes: Collection<String>,
    ): List<Photo>

    // Reconciliation backlog: photos whose async indexing hasn't settled. The
    // cutoff skips photos the AFTER_COMMIT hot path is probably still handling,
    // so the sweep only re-drives genuinely-stuck work. Backed by the partial
    // index idx_photos_indexing_pending (migration V21).
    @Query(
        """
        SELECT p FROM Photo p
        WHERE p.indexingStatus IN :statuses
          AND p.indexingAttempts < :maxAttempts
          AND p.uploadedAt < :cutoff
        ORDER BY p.uploadedAt ASC
        """,
    )
    fun findIndexingBacklog(
        @Param("statuses") statuses: Collection<IndexingStatus>,
        @Param("maxAttempts") maxAttempts: Int,
        @Param("cutoff") cutoff: OffsetDateTime,
        pageable: Pageable,
    ): List<Photo>

    // Phase C batch drain: distinct events that have indexable backlog. The
    // drain groups per event because each mega job is single-event (event
    // isolation — a person enrolled in a mega is stamped with that one event_id).
    @Query(
        """
        SELECT DISTINCT p.eventId FROM Photo p
        WHERE p.indexingStatus IN :statuses
          AND p.indexingAttempts < :maxAttempts
          AND p.uploadedAt < :cutoff
        """,
    )
    fun findEventsWithIndexingBacklog(
        @Param("statuses") statuses: Collection<IndexingStatus>,
        @Param("maxAttempts") maxAttempts: Int,
        @Param("cutoff") cutoff: OffsetDateTime,
        pageable: Pageable,
    ): List<UUID>

    // Orphan reaper — every ai-api person id still referenced by a photo in the
    // event, across ALL statuses (a HIDDEN photo still legitimately owns its
    // person). The reaper deletes any ai-api person for the event NOT in this
    // set. No status filter on purpose: filtering would make live persons look
    // orphaned and get erased.
    @Query(
        """
        SELECT DISTINCT fp.aiPersonId FROM Photo p
        JOIN p.facePersons fp
        WHERE p.eventId = :eventId
        """,
    )
    fun findReferencedAiPersonIds(@Param("eventId") eventId: UUID): List<String>

    // Orphan reaper — distinct events that have at least one indexed face, i.e.
    // the events that could hold ai-api persons worth reconciling. Bounded by
    // the Pageable so one sweep stays cheap.
    @Query(
        """
        SELECT DISTINCT p.eventId FROM Photo p
        JOIN p.facePersons fp
        """,
    )
    fun findEventsWithFacePersons(pageable: Pageable): List<UUID>

    // The PENDING/FAILED photos of ONE event, oldest first, capped by the
    // Pageable to the batch max-size. These get flipped to BATCHING and shipped
    // as a single mega job per kind.
    @Query(
        """
        SELECT p FROM Photo p
        WHERE p.eventId = :eventId
          AND p.indexingStatus IN :statuses
          AND p.indexingAttempts < :maxAttempts
          AND p.uploadedAt < :cutoff
        ORDER BY p.uploadedAt ASC
        """,
    )
    fun findIndexingBacklogForEvent(
        @Param("eventId") eventId: UUID,
        @Param("statuses") statuses: Collection<IndexingStatus>,
        @Param("maxAttempts") maxAttempts: Int,
        @Param("cutoff") cutoff: OffsetDateTime,
        pageable: Pageable,
    ): List<Photo>
}
