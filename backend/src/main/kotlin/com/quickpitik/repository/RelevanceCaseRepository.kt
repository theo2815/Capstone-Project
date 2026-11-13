package com.quickpitik.repository

import com.quickpitik.entity.RelevanceCase
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

// RV-1.1: relevance_cases, written only by service/relevance. Queries come with the tasks that need them.
interface RelevanceCaseRepository : JpaRepository<RelevanceCase, UUID> {

    // RV-1.5 (plan 5.8): up to :sampleSize random photos of the batch not yet in this case's sample, from
    // every photo not trashed and not under removal, whatever its status: a hidden junk batch topped up
    // with good photos must not look clean. Returns the photos added.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_case_photos (case_id, photo_id, round)
        SELECT :caseId, p.id, :round FROM photos p
        WHERE p.event_id = :eventId AND p.photographer_id = :photographerId
          AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
          AND NOT EXISTS (SELECT 1 FROM relevance_case_photos s WHERE s.case_id = :caseId AND s.photo_id = p.id)
        ORDER BY random() LIMIT :sampleSize
    """, nativeQuery = true)
    fun pickSample(
        @Param("caseId") caseId: UUID,
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("round") round: Int,
        @Param("sampleSize") sampleSize: Int,
    ): Int

    @Query(value = "SELECT CAST(count(*) AS int) FROM relevance_case_photos WHERE case_id = :caseId", nativeQuery = true)
    fun countSample(@Param("caseId") caseId: UUID): Int
}
