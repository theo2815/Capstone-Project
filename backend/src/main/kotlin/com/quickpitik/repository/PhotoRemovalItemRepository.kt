package com.quickpitik.repository

import com.quickpitik.entity.PhotoRemovalItem
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface PhotoRemovalItemRepository : JpaRepository<PhotoRemovalItem, UUID> {
    fun findByIdAndRequestId(id: UUID, requestId: UUID): PhotoRemovalItem?
    fun findByPhotoIdSnapshotAndStateInOrderByAdminDecidedAtAscIdAsc(photoIdSnapshot: UUID,
        states: Collection<com.quickpitik.entity.RemovalItemState>): List<PhotoRemovalItem>
    fun findByRequestIdOrderByIdAsc(requestId: UUID): List<PhotoRemovalItem>
    fun findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(requestId: UUID, ownerIdSnapshot: UUID): List<PhotoRemovalItem>
}
