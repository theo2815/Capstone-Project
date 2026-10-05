package com.quickpitik.repository

import com.quickpitik.entity.PhotoRemovalRequest
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface PhotoRemovalRequestRepository : JpaRepository<PhotoRemovalRequest, UUID> {
    fun findByCreationCommandId(creationCommandId: UUID): PhotoRemovalRequest?
}
