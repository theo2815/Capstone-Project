package com.quickpitik.repository

import com.quickpitik.entity.DownloadGrant
import com.quickpitik.entity.DownloadGrantId
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface DownloadGrantRepository : JpaRepository<DownloadGrant, DownloadGrantId> {
    fun findByIdOrderId(orderId: UUID): List<DownloadGrant>
}
