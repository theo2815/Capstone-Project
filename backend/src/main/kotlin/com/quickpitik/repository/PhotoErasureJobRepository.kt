package com.quickpitik.repository

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

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT j FROM PhotoErasureJob j WHERE j.id = :id")
    fun lock(id: UUID): PhotoErasureJob?

    @Query("""SELECT j.id FROM PhotoErasureJob j WHERE j.outcome <> com.quickpitik.dto.photos.PhotoErasureOutcome.COMPLETE
        AND j.nextAttemptAt <= :now AND (j.leaseUntil IS NULL OR j.leaseUntil <= :now)
        ORDER BY j.nextAttemptAt, j.id""")
    fun findDue(now: Instant, pageable: Pageable): List<UUID>
}
