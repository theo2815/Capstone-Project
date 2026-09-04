package com.quickpitik.repository

import com.quickpitik.entity.PhotographerCoupon
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface PhotographerCouponRepository : JpaRepository<PhotographerCoupon, UUID> {
    fun findByCode(code: String): PhotographerCoupon?
    fun existsByCodeAndPhotographerIdNot(code: String, photographerId: UUID): Boolean
}
