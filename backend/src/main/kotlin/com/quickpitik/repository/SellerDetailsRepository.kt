package com.quickpitik.repository

import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface SellerDetailsRepository : JpaRepository<SellerDetails, UUID> {
    fun existsByUserIdAndStatusNot(userId: UUID, status: SellerDetailsStatus): Boolean
}
