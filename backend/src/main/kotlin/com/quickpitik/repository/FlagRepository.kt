package com.quickpitik.repository

import jakarta.persistence.LockModeType
import org.springframework.data.jpa.repository.Lock
import com.quickpitik.entity.Flag
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

interface FlagRepository : JpaRepository<Flag, UUID> {

    @Query(
        """
        SELECT f FROM Flag f
        WHERE (:statusWire IS NULL OR f.statusWire = :statusWire)
          AND (:query = '' OR LOWER(f.reason) LIKE LOWER(CONCAT('%', :query, '%')) OR LOWER(f.note) LIKE LOWER(CONCAT('%', :query, '%')))
        ORDER BY f.createdAt DESC, f.id ASC
        """,
        countQuery = """
        SELECT COUNT(f) FROM Flag f
        WHERE (:statusWire IS NULL OR f.statusWire = :statusWire)
          AND (:query = '' OR LOWER(f.reason) LIKE LOWER(CONCAT('%', :query, '%')) OR LOWER(f.note) LIKE LOWER(CONCAT('%', :query, '%')))
        """,
    )
    fun pageForAdmin(
        @Param("statusWire") statusWire: String?,
        @Param("query") query: String?,
        pageable: Pageable,
    ): Page<Flag>

    @Query("SELECT f.targetId FROM Flag f WHERE f.id = :id AND f.targetKindWire = 'photo'")
    fun findPhotoId(@Param("id") id: UUID): UUID?

    // PE-2.2: the photographer behind a relevance flag, read before any lock (the twin of findPhotoId).
    @Query("SELECT f.targetId FROM Flag f WHERE f.id = :id AND f.targetKindWire = 'user' AND f.reason = :reason")
    fun findRelevanceUserTarget(@Param("id") id: UUID, @Param("reason") reason: String): UUID?

    // PE-2.2: the read behind PayoutRequestService.relevanceHoldOpen, the one home of the hold rule.
    @Query(
        """
        SELECT COUNT(f) > 0 FROM Flag f
        WHERE f.targetKindWire = 'user' AND f.targetId = :userId AND f.reason = :reason
          AND f.statusWire IN ('open', 'escalated') AND f.payoutHoldAt IS NOT NULL
        """,
    )
    fun existsOpenPayoutHold(@Param("userId") userId: UUID, @Param("reason") reason: String): Boolean

    // PE-2.2: open relevance flags for a page of photographers, held or not (the admin payout list).
    @Query(
        """
        SELECT f FROM Flag f
        WHERE f.targetKindWire = 'user' AND f.targetId IN :userIds AND f.reason = :reason
          AND f.statusWire IN ('open', 'escalated')
        """,
    )
    fun findOpenByUserTargetsAndReason(
        @Param("userIds") userIds: Collection<UUID>,
        @Param("reason") reason: String,
    ): List<Flag>

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT f FROM Flag f WHERE f.id = :id")
    fun findByIdForUpdate(@Param("id") id: UUID): Flag?

    fun countByStatusWire(statusWire: String): Long

    // SystemFlagService's duplicate lookup. Derived (no @Query): no null bind can hit the PG 18 typing trap.
    fun findByTargetKindWireAndTargetIdAndReasonAndStatusWireIn(
        targetKindWire: String,
        targetId: UUID,
        reason: String,
        statusWires: Collection<String>,
    ): List<Flag>

    // AA-R9: accounts with a live report on one target (the hide threshold).
    @Query(
        """
        SELECT COUNT(DISTINCT f.reporterId) FROM Flag f
        WHERE f.targetKindWire = :targetKind AND f.targetId = :targetId
          AND f.reason = 'user_report' AND f.statusWire IN ('open', 'escalated')
        """,
    )
    fun countDistinctReporters(@Param("targetKind") targetKind: String, @Param("targetId") targetId: UUID): Long

    // PE-2.3 (R2-07): per event, the photos this flag hid that are not restored (flag_hidden_photos, PE-2.4).
    @Query(
        value = """
        SELECT p.event_id AS "eventId", count(*) AS "hidden"
        FROM flag_hidden_photos h JOIN photos p ON p.id = h.photo_id
        WHERE h.flag_id = :flagId AND h.restored_at IS NULL
        GROUP BY p.event_id
        """,
        nativeQuery = true,
    )
    fun countHiddenPhotosByEvent(@Param("flagId") flagId: UUID): List<HiddenPhotoCount>

    // PE-2.4 (plan 5.12): the record of what a relevance flag's batch hide took down. Written only by
    // service/admin. Every non-deleted photo of the batch goes in, whatever its status, so a photo a
    // single-photo flag already hid is held by this flag too. DO NOTHING (R2-07): a second hide never
    // re-hides a photo an admin restored.
    @Modifying
    @Query(
        value = """
        INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at)
        SELECT :flagId, p.id, :now FROM photos p
        WHERE p.event_id = :eventId AND p.photographer_id = :photographerId AND p.deleted_at IS NULL
        ON CONFLICT (flag_id, photo_id) DO NOTHING
        """,
        nativeQuery = true,
    )
    fun recordBatchHide(
        @Param("flagId") flagId: UUID,
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("now") now: OffsetDateTime,
    ): Int

    // Sorted, so the caller can lock the photo rows in id order before it reads the restore rule.
    @Query(
        value = "SELECT photo_id FROM flag_hidden_photos WHERE flag_id = :flagId AND restored_at IS NULL ORDER BY photo_id",
        nativeQuery = true,
    )
    fun unrestoredPhotoIds(@Param("flagId") flagId: UUID): List<UUID>

    @Query(
        value = "SELECT count(*) > 0 FROM flag_hidden_photos WHERE flag_id = :flagId AND photo_id = :photoId",
        nativeQuery = true,
    )
    fun isRecorded(@Param("flagId") flagId: UUID, @Param("photoId") photoId: UUID): Boolean

    @Modifying
    @Query(
        value = "UPDATE flag_hidden_photos SET restored_at = :now WHERE flag_id = :flagId AND photo_id = :photoId AND restored_at IS NULL",
        nativeQuery = true,
    )
    fun markRestored(@Param("flagId") flagId: UUID, @Param("photoId") photoId: UUID, @Param("now") now: OffsetDateTime): Int

    @Modifying
    @Query(
        value = "UPDATE flag_hidden_photos SET restored_at = :now WHERE flag_id = :flagId AND restored_at IS NULL",
        nativeQuery = true,
    )
    fun markAllRestored(@Param("flagId") flagId: UUID, @Param("now") now: OffsetDateTime): Int

    // Half of AdminFlagService.stillHeldHidden: another flag's batch hide still holds this photo.
    @Query(
        value = """
        SELECT count(*) > 0 FROM flag_hidden_photos
        WHERE photo_id = :photoId AND flag_id <> :flagId AND restored_at IS NULL
        """,
        nativeQuery = true,
    )
    fun existsOtherBatchHold(@Param("photoId") photoId: UUID, @Param("flagId") flagId: UUID): Boolean

    // Restored rows stay in the list (the admin sees what was put back); oldest hide first, ties by photo id.
    @Query(
        value = """
        SELECT h.photo_id AS "photoId", p.event_id AS "eventId",
               COALESCE(p.watermark_s3_key, p.thumbnail_s3_key) AS "thumbnailKey",
               h.hidden_at AS "hiddenAt", h.restored_at AS "restoredAt"
        FROM flag_hidden_photos h JOIN photos p ON p.id = h.photo_id
        WHERE h.flag_id = :flagId
        ORDER BY h.hidden_at, h.photo_id
        """,
        countQuery = "SELECT count(*) FROM flag_hidden_photos WHERE flag_id = :flagId",
        nativeQuery = true,
    )
    fun pageHiddenPhotos(@Param("flagId") flagId: UUID, pageable: Pageable): Page<HiddenPhotoRow>

    fun existsByTargetKindWireAndTargetIdAndStatusWireAndIdNot(
        targetKindWire: String,
        targetId: UUID,
        statusWire: String,
        id: UUID,
    ): Boolean
}

interface HiddenPhotoRow {
    val photoId: UUID
    val eventId: UUID
    val thumbnailKey: String?
    val hiddenAt: java.time.Instant
    val restoredAt: java.time.Instant?
}

interface HiddenPhotoCount {
    val eventId: UUID
    val hidden: Long
}
