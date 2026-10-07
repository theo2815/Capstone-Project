package com.quickpitik.repository

import com.quickpitik.entity.IndexingStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

interface PhotoWorkSnapshot {
    val token: UUID
    val photoId: UUID
    val generation: Long
    val kind: String
    val expiresAt: java.time.Instant
}

interface PhotoFaceWork {
    val token: UUID
    val eventId: UUID
    val photoId: UUID
    val generation: Long
    val expiresAt: java.time.Instant
    val provider: String
    val personName: String?
    val personIds: String
    val priorProvider: String?
    val priorPersonIds: String
    val jobIds: String
    val batchPhotoIds: String
    val finishedAt: java.time.Instant?
    val facesErasedAt: java.time.Instant?
}

interface PhotoFaceTarget {
    val eventId: UUID
    val photoId: UUID
}

interface PhotoRepository : JpaRepository<Photo, UUID> {

    // Scalar read bypasses an already-hydrated entity immediately before delivery.
    @Query(value = "SELECT EXISTS (SELECT 1 FROM photos WHERE id = :id AND removal_state <> 'REMOVED' AND deletion_started_at IS NULL AND purged_at IS NULL)", nativeQuery = true)
    fun isDeliveryAvailable(@Param("id") id: UUID): Boolean

    // Public/free streaming also requires the current catalog predicate, not paid access.
    @Query(value = """
        SELECT EXISTS (SELECT 1 FROM photos p JOIN events e ON e.id = p.event_id
            WHERE p.id = :photoId AND p.event_id = :eventId AND p.status = 'LIVE'
            AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
            AND p.deletion_started_at IS NULL AND p.purged_at IS NULL
            AND e.pricing_mode = 'FREE' AND e.status NOT IN ('DRAFT', 'ARCHIVED'))
    """, nativeQuery = true)
    fun isFreeDeliveryAvailable(eventId: UUID, photoId: UUID): Boolean

    @Query(value = """
        SELECT EXISTS (SELECT 1 FROM photos WHERE event_id = :eventId AND photographer_id = :photographerId
            AND deleted_at IS NULL AND deletion_started_at IS NULL AND purged_at IS NULL AND removal_state <> 'REMOVED')
    """, nativeQuery = true)
    fun hasDeliverableOwnerPhoto(eventId: UUID, photographerId: UUID): Boolean

    @Query(value = "SELECT EXISTS (SELECT 1 FROM photos WHERE id = :id AND removal_state = 'REMOVED')", nativeQuery = true)
    fun isRemoved(@Param("id") id: UUID): Boolean

    @Query("SELECT p.removalState FROM Photo p WHERE p.id = :id")
    fun findCurrentRemovalState(@Param("id") id: UUID): com.quickpitik.entity.RemovalState?

    @Query("""
        SELECT DISTINCT p FROM Photo p LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId AND (:bib = '' OR UPPER(b.bibNumber) LIKE CONCAT('%', UPPER(:bib), '%'))
        ORDER BY p.capturedAt DESC NULLS LAST, p.uploadedAt DESC, p.id ASC
    """, countQuery = """
        SELECT COUNT(DISTINCT p) FROM Photo p LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId AND (:bib = '' OR UPPER(b.bibNumber) LIKE CONCAT('%', UPPER(:bib), '%'))
    """)
    fun searchForRemovalReview(eventId: UUID, bib: String, pageable: Pageable): Page<Photo>

    @Query("""
        SELECT DISTINCT p FROM Photo p JOIN p.facePersons fp
        WHERE p.eventId = :eventId AND fp.aiPersonId IN :personIds
        ORDER BY p.capturedAt DESC NULLS LAST, p.uploadedAt DESC, p.id ASC
    """)
    fun findRemovalReviewCandidates(eventId: UUID, personIds: Collection<String>): List<Photo>

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    // DISTINCT forces follow-on locking: checkout could hydrate availability
    // before waiting for a review's lock. Lock the row in the initial read.
    @Query("SELECT p FROM Photo p WHERE p.id IN :ids ORDER BY p.id")
    fun findAllByIdForUpdate(@Param("ids") ids: Collection<UUID>): List<Photo>

    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query(value = """
        UPDATE photos p
        SET work_generation = work_generation + 1, removal_state = CASE WHEN EXISTS (
            SELECT 1 FROM photo_removal_items i
            WHERE i.photo_id_snapshot = p.id AND i.state IN ('VERIFIED', 'RETAINED_RESTRICTED')
        ) THEN 'REVIEW' ELSE 'NONE' END
        WHERE p.id IN :ids AND p.removal_state <> 'REMOVED'
          AND p.removal_state <> CASE WHEN EXISTS (
            SELECT 1 FROM photo_removal_items i
            WHERE i.photo_id_snapshot = p.id AND i.state IN ('VERIFIED', 'RETAINED_RESTRICTED')
          ) THEN 'REVIEW' ELSE 'NONE' END
        """, nativeQuery = true)
    fun refreshRemovalState(@Param("ids") ids: Collection<UUID>): Int

    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query("UPDATE Photo p SET p.removalState = com.quickpitik.entity.RemovalState.REMOVED, p.workGeneration = p.workGeneration + 1 " +
        "WHERE p.id = :id AND p.removalState <> com.quickpitik.entity.RemovalState.REMOVED")
    fun authorizeRemoval(@Param("id") id: UUID): Int

    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query(value = """UPDATE photos SET
        watermark_work_token = CASE WHEN :kind = 'WATERMARK' THEN :token ELSE watermark_work_token END,
        watermark_work_expires_at = CASE WHEN :kind = 'WATERMARK' THEN :expiry ELSE watermark_work_expires_at END,
        indexing_work_token = CASE WHEN :kind = 'INDEXING' THEN :token ELSE indexing_work_token END,
        indexing_work_expires_at = CASE WHEN :kind = 'INDEXING' THEN :expiry ELSE indexing_work_expires_at END
        WHERE id = :id AND removal_state = 'NONE' AND deleted_at IS NULL AND deletion_started_at IS NULL
          AND CASE WHEN :kind = 'WATERMARK' THEN watermark_work_token ELSE indexing_work_token END IS NULL""", nativeQuery = true)
    fun claimWork(id: UUID, kind: String, token: UUID, expiry: OffsetDateTime): Int

    @Modifying
    @Query(value = """INSERT INTO photo_work_outputs(token, photo_id, event_id, generation, kind, provider,
        expires_at, storage_key, person_name, prior_provider, prior_person_ids) VALUES
        (:token, :photoId, :eventId, :generation, :kind, :provider, :expiry, :key, :personName, :priorProvider, CAST(:priorPersons AS jsonb))""", nativeQuery = true)
    fun recordWork(token: UUID, photoId: UUID, eventId: UUID, generation: Long, kind: String,
        provider: String, expiry: OffsetDateTime, key: String?, personName: String?, priorProvider: String?, priorPersons: String): Int

    @Modifying
    @Query(value = "UPDATE photo_work_outputs SET person_ids = person_ids || CAST(:persons AS jsonb), faces_erased_at = NULL WHERE token = :token", nativeQuery = true)
    fun recordWorkPersons(token: UUID, persons: String): Int

    @Query(value = """SELECT w.token, w.event_id AS "eventId", w.photo_id AS "photoId", w.generation,
        w.expires_at AS "expiresAt", w.provider, w.person_name AS "personName", w.person_ids::text AS "personIds",
        w.prior_provider AS "priorProvider", w.prior_person_ids::text AS "priorPersonIds", w.job_ids::text AS "jobIds",
        COALESCE((SELECT jsonb_agg(bp.photo_id ORDER BY bp.idx)::text FROM ai_index_batch_photos bp
            WHERE bp.batch_id = w.batch_id), '[]') AS "batchPhotoIds",
        w.finished_at AS "finishedAt", w.faces_erased_at AS "facesErasedAt"
        FROM photo_work_outputs w WHERE w.event_id = :eventId AND w.photo_id = :photoId AND w.kind = 'INDEXING'
        ORDER BY w.created_at, w.token""", nativeQuery = true)
    fun faceWorkForPhoto(eventId: UUID, photoId: UUID): List<PhotoFaceWork>

    @Query(value = """SELECT DISTINCT w.event_id AS "eventId", w.photo_id AS "photoId"
        FROM photo_work_outputs w WHERE w.kind = 'INDEXING' AND w.faces_erased_at IS NULL
        AND NOT EXISTS (SELECT 1 FROM photos p WHERE p.id = w.photo_id
            AND p.removal_state <> 'REMOVED' AND p.deletion_started_at IS NULL)
        ORDER BY w.event_id, w.photo_id""", nativeQuery = true)
    fun faceWorkPending(pageable: Pageable): List<PhotoFaceTarget>

    @Modifying
    @Query(value = """UPDATE photo_work_outputs SET faces_erased_at = :now
        WHERE token = :token AND finished_at IS NOT NULL AND person_ids = CAST(:persons AS jsonb)""", nativeQuery = true)
    fun confirmFaceWorkErased(token: UUID, persons: String, now: OffsetDateTime): Int

    @Modifying
    @Query(value = """UPDATE photo_work_outputs SET
        provider = CASE WHEN :prior THEN provider ELSE :provider END,
        prior_provider = CASE WHEN :prior THEN :provider ELSE prior_provider END
        WHERE token = :token AND event_id = :eventId AND photo_id = :photoId AND kind = 'INDEXING'
          AND person_name = CAST(:photoId AS text)
          AND COALESCE(CASE WHEN :prior THEN prior_provider ELSE provider END, 'unknown') IN ('unknown', 'UNKNOWN')
          AND CASE WHEN :prior THEN prior_person_ids ELSE person_ids END = CAST(:persons AS jsonb)""", nativeQuery = true)
    fun attributeFaceWorkProvider(token: UUID, eventId: UUID, photoId: UUID, prior: Boolean, persons: String, provider: String): Int

    @Modifying
    @Query(value = "UPDATE photo_work_outputs SET batch_id = :batchId WHERE token = :token", nativeQuery = true)
    fun attributeWorkBatch(token: UUID, batchId: UUID): Int

    @Modifying
    @Query(value = "UPDATE photo_work_outputs SET job_ids = job_ids || CAST(:jobs AS jsonb) WHERE token = :token", nativeQuery = true)
    fun recordWorkJob(token: UUID, jobs: String): Int

    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query(value = """WITH renewed AS (
        UPDATE photos SET indexing_work_expires_at = :expiry
        WHERE id = :photoId AND indexing_work_token = :token RETURNING id
        ) UPDATE photo_work_outputs SET expires_at = :expiry
        WHERE token = :token AND finished_at IS NULL AND EXISTS (SELECT 1 FROM renewed)""", nativeQuery = true)
    fun renewIndexingWork(photoId: UUID, token: UUID, expiry: OffsetDateTime): Int

    @Modifying
    @Query(value = "UPDATE photo_work_outputs SET finished_at = :now WHERE token = :token AND finished_at IS NULL", nativeQuery = true)
    fun finishWorkOutput(token: UUID, now: OffsetDateTime): Int

    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query(value = """UPDATE photos SET
        watermark_work_token = CASE WHEN watermark_work_token = :token THEN NULL ELSE watermark_work_token END,
        watermark_work_expires_at = CASE WHEN watermark_work_token = :token THEN NULL ELSE watermark_work_expires_at END,
        indexing_work_token = CASE WHEN indexing_work_token = :token THEN NULL ELSE indexing_work_token END,
        indexing_work_expires_at = CASE WHEN indexing_work_token = :token THEN NULL ELSE indexing_work_expires_at END
        WHERE id = :id""", nativeQuery = true)
    fun finishWork(id: UUID, token: UUID): Int

    @Query(value = """SELECT token, photo_id AS "photoId", generation, kind, expires_at AS "expiresAt"
        FROM photo_work_outputs WHERE batch_id = :batchId ORDER BY photo_id""", nativeQuery = true)
    fun workForBatch(batchId: UUID): List<PhotoWorkSnapshot>

    @Query(value = """SELECT EXISTS (SELECT 1 FROM photo_work_outputs
        WHERE photo_id = :photoId AND finished_at IS NULL AND (:kind = 'ANY' OR kind = :kind))""", nativeQuery = true)
    fun hasUnfinishedWork(photoId: UUID, kind: String): Boolean

    @Modifying
    @Query("UPDATE Photo p SET p.deletedAt = :now WHERE p.photographerId = :photographerId " +
        "AND p.eventId = :eventId AND p.id IN :ids AND p.deletedAt IS NULL")
    fun trashOwned(@Param("photographerId") photographerId: UUID, @Param("eventId") eventId: UUID,
        @Param("ids") ids: Collection<UUID>, @Param("now") now: OffsetDateTime): Int

    @Modifying
    @Query("UPDATE Photo p SET p.deletedAt = NULL WHERE p.photographerId = :photographerId " +
        "AND p.eventId = :eventId AND p.id IN :ids AND p.deletedAt IS NOT NULL AND p.deletionStartedAt IS NULL")
    fun restoreOwned(@Param("photographerId") photographerId: UUID, @Param("eventId") eventId: UUID,
        @Param("ids") ids: Collection<UUID>): Int

    @Modifying
    @Query("UPDATE Photo p SET p.deletionStartedAt = :now WHERE p.id = :id " +
        "AND p.deletedAt IS NOT NULL AND p.deletionStartedAt IS NULL")
    fun startDeletion(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    @Modifying
    @Query("UPDATE Photo p SET p.indexingStatus = com.quickpitik.entity.IndexingStatus.SKIPPED " +
        "WHERE p.id = :id AND p.deletedAt IS NULL " +
        "AND p.indexingStatus <> com.quickpitik.entity.IndexingStatus.INDEXED")
    fun skipIndexing(@Param("id") id: UUID): Int

    // Bulk re-prices every photo under the event. Used by admin event PATCH
    // when the operator changes events.price_per_photo so existing photos
    // pick up the new price across runner-facing galleries. Carts follow
    // automatically: CartService renders the live photos.price_php and
    // OrderService.create charges it, so cart_items.price_php_at_add is
    // deliberately left alone — nothing displays or bills off that column.
    // V46: an approved watermark-policy change makes every LIVE preview stale.
    // Back to PROCESSING with a fresh budget; PhotoWatermarkTrigger.reconcile
    // re-renders them under the new policy within a sweep. Runners lose the
    // photos for that cycle — acceptable for an owner-requested, admin-
    // approved change.
    @Modifying
    @Query(
        "UPDATE Photo p SET p.status = com.quickpitik.entity.PhotoStatus.PROCESSING, p.processingAttempts = 0 " +
            "WHERE p.eventId = :eventId AND p.status = com.quickpitik.entity.PhotoStatus.LIVE " +
            "AND p.removalState = com.quickpitik.entity.RemovalState.NONE",
    )
    fun resetForRewatermark(@Param("eventId") eventId: UUID): Int

    @Modifying
    @Query("UPDATE Photo p SET p.pricePhp = :price WHERE p.eventId = :eventId")
    fun updatePriceByEventId(
        @Param("eventId") eventId: UUID,
        @Param("price") price: BigDecimal,
    ): Int

    // Bulk re-queue for AI indexing (admin reindex endpoint). Resets the
    // attempt budget so the reconcile sweep actually picks the rows up —
    // exhausted attempts are otherwise excluded forever.
    @Modifying
    @Query(
        "UPDATE Photo p SET p.indexingStatus = com.quickpitik.entity.IndexingStatus.PENDING, " +
            "p.indexingAttempts = 0, p.indexingError = null " +
            "WHERE p.eventId = :eventId AND p.indexingStatus IN :statuses " +
            "AND p.removalState = com.quickpitik.entity.RemovalState.NONE",
    )
    fun requeueIndexing(
        @Param("eventId") eventId: UUID,
        @Param("statuses") statuses: Collection<IndexingStatus>,
    ): Int

    // Async-watermark reconcile sweep (V36): PROCESSING rows the AFTER_COMMIT
    // hot path missed (app crash, storage outage), oldest first. Mirrors
    // findIndexingBacklog.
    @Query(
        """
        SELECT p FROM Photo p
        WHERE p.status = com.quickpitik.entity.PhotoStatus.PROCESSING
          AND p.deletedAt IS NULL
          AND p.deletionStartedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
          AND p.watermarkWorkToken IS NULL
          AND p.processingAttempts < :maxAttempts
          AND p.uploadedAt < :cutoff
        ORDER BY p.uploadedAt ASC
        """,
    )
    fun findWatermarkBacklog(
        @Param("maxAttempts") maxAttempts: Int,
        @Param("cutoff") cutoff: OffsetDateTime,
        pageable: Pageable,
    ): List<Photo>

    // Targeted, conditional flip to LIVE once the watermark object is stored.
    // Deliberately NOT an entity save: PhotoIndexingService writes the same row
    // concurrently. Touching only the watermark columns and @DynamicUpdate on
    // Photo keep the two writers column-disjoint; the status guard makes
    // re-runs no-ops.
    @Modifying
    @Query(
        "UPDATE Photo p SET p.watermarkS3Key = :key, p.thumbnailS3Key = :key, p.phash = :phash, " +
            "p.phashClean = :phashClean, p.phashCentre = :phashCentre, p.publishedAt = CURRENT_TIMESTAMP, " +
            "p.status = com.quickpitik.entity.PhotoStatus.LIVE " +
            "WHERE p.id = :id AND p.status = com.quickpitik.entity.PhotoStatus.PROCESSING AND p.deletedAt IS NULL " +
            "AND p.deletionStartedAt IS NULL AND p.removalState = com.quickpitik.entity.RemovalState.NONE",
    )
    fun publishWatermarked(
        @Param("id") id: UUID,
        @Param("key") key: String,
        @Param("phash") phash: Long,
        @Param("phashClean") phashClean: Long,
        @Param("phashCentre") phashCentre: Long,
    ): Int

    @Modifying
    @Query("UPDATE Photo p SET p.processingAttempts = p.processingAttempts + 1 WHERE p.id = :id")
    fun incrementProcessingAttempts(@Param("id") id: UUID): Int

    // Admin flag queue. Conditional for the same reason as publishWatermarked:
    // the watermark pipeline writes this row concurrently, so a load-modify-save
    // from the flag queue would clobber the columns it just set. Already-HIDDEN
    // rows are a no-op.
    @Modifying
    @Query(
        "UPDATE Photo p SET p.status = com.quickpitik.entity.PhotoStatus.HIDDEN " +
            "WHERE p.id = :id AND p.status IN (com.quickpitik.entity.PhotoStatus.LIVE, " +
            "com.quickpitik.entity.PhotoStatus.PROCESSING)",
    )
    fun hideForFlag(@Param("id") id: UUID): Int

    // Restore. The target is the caller's call (see AdminFlagService): LIVE only
    // when a derived asset exists, PROCESSING otherwise. processingAttempts
    // resets so a budget exhausted before the hide can't strand the row outside
    // findWatermarkBacklog.
    @Modifying
    @Query(
        "UPDATE Photo p SET p.status = :target, p.processingAttempts = 0 " +
            "WHERE p.id = :id AND p.status = com.quickpitik.entity.PhotoStatus.HIDDEN",
    )
    fun unhideForFlag(@Param("id") id: UUID, @Param("target") target: PhotoStatus): Int

    // Nearest stored fingerprint by Hamming distance, for POST
    // /public/photos/verify. The upload is compared against all three
    // registered hashes — marked frame, clean frame, clean centre crop (V43;
    // pre-V43 rows have only the first, and a NULL loses at 64) — so a
    // screenshot, a cleaned copy and a crop to the runner all attribute.
    // Returns [photographer_id, event_id, distance] — deliberately no photo
    // id, so the answer can never become a photo URL.
    // ponytail: full scan of LIVE rows; a BK-tree or pg extension when photos > ~1M.
    // `LIVE, not trashed, removal NONE` = Photo.isSellable — JPQL/SQL cannot call the
    // Kotlin helper, so this copy changes with it.
    @Query(
        value = """
        SELECT p.photographer_id, p.event_id,
               LEAST(
                 bit_count(CAST((p.phash # :h) AS bit(64))),
                 COALESCE(bit_count(CAST((p.phash_clean # :h) AS bit(64))), 64),
                 COALESCE(bit_count(CAST((p.phash_centre # :h) AS bit(64))), 64)
               ) AS d
        FROM photos p
        WHERE p.status = 'LIVE' AND p.deleted_at IS NULL AND p.removal_state = 'NONE' AND p.phash IS NOT NULL
        ORDER BY d ASC
        LIMIT 1
        """,
        nativeQuery = true,
    )
    fun findNearestByPhash(@Param("h") h: Long): List<Array<Any>>

    // No-bib event grid: merge each photographer's first-ranked photo, then
    // each second-ranked photo, and so on. The snapshot freezes the LIVE set
    // while the visitor pages; the seeded hash only varies near-equal rows.
    // ponytail: this window ranks the event's LIVE rows; materialize only if
    // EXPLAIN shows it hurting at real event scale.
    // `LIVE, not trashed, removal NONE` = Photo.isSellable — JPQL/SQL cannot call the
    // Kotlin helper, so this copy changes with it.
    @Query(
        value = """
        WITH ranked AS (
            SELECT p.id,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.photographer_id
                       ORDER BY p.captured_at DESC NULLS LAST, p.uploaded_at DESC, p.id ASC
                   ) AS photographer_round,
                   DATE_TRUNC('hour', COALESCE(p.captured_at, p.uploaded_at)) AS relevance_bucket,
                   p.photographer_id,
                   p.captured_at,
                   p.uploaded_at
            FROM photos p
            WHERE p.event_id = :eventId
              AND p.status = 'LIVE'
              AND p.deleted_at IS NULL
              AND p.removal_state = 'NONE'
              AND p.published_at <= :snapshotAt
        )
        SELECT p.*
        FROM ranked r
        JOIN photos p ON p.id = r.id
        ORDER BY r.photographer_round ASC,
                 (r.photographer_id IS NULL) ASC,
                 r.relevance_bucket DESC,
                 hashtextextended(COALESCE(r.photographer_id::text, 'unknown'), :seed) ASC,
                 r.captured_at DESC NULLS LAST,
                 r.uploaded_at DESC,
                 r.id ASC
        """,
        countQuery = """
        SELECT COUNT(*) FROM photos p
        WHERE p.event_id = :eventId
          AND p.status = 'LIVE'
          AND p.deleted_at IS NULL
          AND p.removal_state = 'NONE'
          AND p.published_at <= :snapshotAt
        """,
        nativeQuery = true,
    )
    fun findForEventNoBib(
        @Param("eventId") eventId: UUID,
        @Param("snapshotAt") snapshotAt: OffsetDateTime,
        @Param("seed") seed: Long,
        pageable: Pageable,
    ): Page<Photo>

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
          AND p.deletedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
          AND (:bib = '' OR UPPER(b.bibNumber) LIKE CONCAT('%', UPPER(:bib), '%'))
        ORDER BY p.capturedAt DESC NULLS LAST, p.uploadedAt DESC, p.id ASC
        """,
        countQuery = """
        SELECT COUNT(DISTINCT p) FROM Photo p
        LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND p.deletedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
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
          AND p.deletedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
          AND fp.aiPersonId IN :aiPersonIds
        ORDER BY p.capturedAt DESC NULLS LAST, p.uploadedAt DESC, p.id ASC
        """,
        countQuery = """
        SELECT COUNT(DISTINCT p) FROM Photo p
        JOIN p.facePersons fp
        WHERE p.eventId = :eventId
          AND p.status = :status
          AND p.deletedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
          AND fp.aiPersonId IN :aiPersonIds
        """,
    )
    fun findByEventAndPersonIds(
        @Param("eventId") eventId: UUID,
        @Param("status") status: PhotoStatus,
        @Param("aiPersonIds") aiPersonIds: Collection<String>,
        pageable: Pageable,
    ): Page<Photo>

    // Photographer-scoped ACTIVE library: every non-trashed photo in the event
    // uploaded by this photographer, regardless of lifecycle status (the
    // photographer can manage HIDDEN/PROCESSING rows from their dashboard).
    // Trashed rows (deleted_at set) are excluded — see findPhotographerTrash.
    // Sort comes from the Pageable so the caller can flip newest|oldest.
    @Query(
        """
        SELECT p FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
          AND p.deletedAt IS NULL
        """,
        countQuery = """
        SELECT COUNT(p) FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
          AND p.deletedAt IS NULL
        """,
    )
    fun findPhotographerLibrary(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        pageable: Pageable,
    ): Page<Photo>

    // The photographer's Deleted Photos for an event: only trashed rows
    // (deleted_at set). Mirrors findPhotographerLibrary so the same DTO mapper
    // + pagination serve both the active tab and the trash tab.
    @Query(
        """
        SELECT p FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
          AND p.deletedAt IS NOT NULL
        """,
        countQuery = """
        SELECT COUNT(p) FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
          AND p.deletedAt IS NOT NULL
        """,
    )
    fun findPhotographerTrash(
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
          AND p.deletedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
        """,
        countQuery = """
        SELECT COUNT(p) FROM Photo p
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
          AND p.status = :status
          AND p.deletedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
        """,
    )
    fun findPhotographerLibraryByStatus(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("status") status: PhotoStatus,
        pageable: Pageable,
    ): Page<Photo>

    // Public catalog counts use the same predicate as Photo.isSellable;
    // upload/owner/admin counters retain their existing meaning.
    @Query("SELECT COUNT(p) FROM Photo p WHERE p.eventId = :eventId " +
        "AND p.status = com.quickpitik.entity.PhotoStatus.LIVE AND p.deletedAt IS NULL " +
        "AND p.removalState = com.quickpitik.entity.RemovalState.NONE")
    fun countPublicForEvent(@Param("eventId") eventId: UUID): Long

    @Query("SELECT p.eventId AS eventId, COUNT(p) AS photoCount FROM Photo p WHERE p.eventId IN :eventIds " +
        "AND p.status = com.quickpitik.entity.PhotoStatus.LIVE AND p.deletedAt IS NULL " +
        "AND p.removalState = com.quickpitik.entity.RemovalState.NONE GROUP BY p.eventId")
    fun countPublicByEventIds(@Param("eventIds") eventIds: Collection<UUID>): List<PublicEventPhotoCount>

    @Query("SELECT p.eventId AS eventId, COUNT(p) AS photoCount FROM Photo p WHERE p.eventId IN :eventIds " +
        "AND p.photographerId = :photographerId " +
        "AND p.status = com.quickpitik.entity.PhotoStatus.LIVE AND p.deletedAt IS NULL " +
        "AND p.removalState = com.quickpitik.entity.RemovalState.NONE GROUP BY p.eventId")
    fun countPublicForPhotographerByEventIds(@Param("eventIds") eventIds: Collection<UUID>,
        @Param("photographerId") photographerId: UUID): List<PublicEventPhotoCount>

    fun findFirstByIdAndPhotographerId(id: UUID, photographerId: UUID): Photo?

    fun findByIdAndPhotographerIdAndDeletionStartedAtIsNull(id: UUID, photographerId: UUID): Photo?

    fun existsByS3Key(s3Key: String): Boolean

    // Owner+event-scoped bulk load for trash / restore / permanent-delete. A
    // foreign or wrong-event id simply isn't returned, so a bulk operation can
    // never touch another photographer's (or another event's) photos.
    fun findAllByIdInAndPhotographerIdAndEventId(
        ids: Collection<UUID>,
        photographerId: UUID,
        eventId: UUID,
    ): List<Photo>

    // Guard for the "Download All" prepare: is there anything active to zip?
    // (Also proves coverage, since it is photographer-scoped.)
    fun countByEventIdAndPhotographerIdAndDeletedAtIsNull(eventId: UUID, photographerId: UUID): Long

    // Lightweight rows for the "Download All" ZIP: the original key + the
    // alphabetically-first bib per active photo. A projection, not entities, so
    // a large event never loads thousands of Photos (with their eager bib/face
    // collections) into memory just to name ZIP entries.
    @Query(
        """
        SELECT p.id AS id, p.s3Key AS s3Key, MIN(b.bibNumber) AS bib
        FROM Photo p
        LEFT JOIN p.bibs b
        WHERE p.eventId = :eventId
          AND p.photographerId = :photographerId
          AND p.deletedAt IS NULL
        GROUP BY p.id, p.s3Key
        ORDER BY p.id ASC
        """,
    )
    fun findActiveBundleRows(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
    ): List<PhotoBundleRow>

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
          AND p.deletedAt IS NULL
          AND p.deletionStartedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
          AND p.indexingWorkToken IS NULL
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
          AND p.deletedAt IS NULL
          AND p.deletionStartedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
          AND p.indexingWorkToken IS NULL
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
          AND p.deletedAt IS NULL
          AND p.deletionStartedAt IS NULL
          AND p.removalState = com.quickpitik.entity.RemovalState.NONE
          AND p.indexingWorkToken IS NULL
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

// Projection for the photographer "Download All" ZIP — see findActiveBundleRows.
interface PhotoBundleRow {
    val id: UUID
    val s3Key: String
    val bib: String?
}

interface PublicEventPhotoCount {
    val eventId: UUID
    val photoCount: Long
}
