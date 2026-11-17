package com.quickpitik.repository

import com.quickpitik.dto.photos.PhotoErasureOutcome
import com.quickpitik.entity.PhotoErasureJob
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import java.time.Instant
import java.util.UUID

interface PhotoErasureJobRepository : JpaRepository<PhotoErasureJob, UUID> {
    fun findByPhotoId(photoId: UUID): PhotoErasureJob?
    fun countByOutcomeNot(outcome: PhotoErasureOutcome): Long

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT j FROM PhotoErasureJob j WHERE j.id = :id")
    fun lock(id: UUID): PhotoErasureJob?

    // Privacy removals first (few, legally urgent), then fewer attempts, so a backlog of retried deletes cannot starve new jobs (RD-17 F1).
    @Query("""SELECT j.id FROM PhotoErasureJob j WHERE j.outcome <> com.quickpitik.dto.photos.PhotoErasureOutcome.COMPLETE
        AND j.nextAttemptAt <= :now AND (j.leaseUntil IS NULL OR j.leaseUntil <= :now)
        ORDER BY CASE WHEN j.purpose = com.quickpitik.dto.photos.PhotoErasurePurpose.PRIVACY_REMOVAL THEN 0 ELSE 1 END,
            j.attempts, j.nextAttemptAt, j.id""")
    fun findDue(now: Instant, pageable: Pageable): List<UUID>
}
