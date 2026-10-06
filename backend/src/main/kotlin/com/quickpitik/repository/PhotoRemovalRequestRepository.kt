package com.quickpitik.repository

import com.quickpitik.entity.PhotoRemovalRequest
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.JpaSpecificationExecutor
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.domain.Pageable
import jakarta.persistence.LockModeType
import java.time.Instant
import java.util.UUID

interface PhotoRemovalRequestRepository : JpaRepository<PhotoRemovalRequest, UUID>, JpaSpecificationExecutor<PhotoRemovalRequest> {
    fun findByCreationCommandId(creationCommandId: UUID): PhotoRemovalRequest?

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT r FROM PhotoRemovalRequest r WHERE r.id = :id")
    fun findForUpdate(id: UUID): PhotoRemovalRequest?

    fun findByEvidenceCleanupNextAttemptAtLessThanEqualOrderByEvidenceCleanupNextAttemptAtAscIdAsc(now: Instant, pageable: Pageable): List<PhotoRemovalRequest>
}
