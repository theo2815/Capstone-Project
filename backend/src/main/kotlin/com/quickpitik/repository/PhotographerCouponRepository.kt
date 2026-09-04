package com.quickpitik.repository

import com.quickpitik.entity.PhotographerCoupon
import jakarta.persistence.LockModeType
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

interface PhotographerCouponRepository : JpaRepository<PhotographerCoupon, UUID> {
    fun findByEventIdAndPhotographerId(eventId: UUID, photographerId: UUID): PhotographerCoupon?
    fun findByCodeAndEventIdIsNotNull(code: String): PhotographerCoupon?
    fun existsByCodeAndEventIdIsNotNullAndIdNot(code: String, id: UUID): Boolean

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT c FROM PhotographerCoupon c WHERE c.code = :code AND c.eventId IS NOT NULL")
    fun findScopedByCodeForUpdate(@Param("code") code: String): PhotographerCoupon?

    @Query(
        value = """
        SELECT pc.*
        FROM photographer_coupons pc
        WHERE pc.event_id = :eventId
          AND pc.photographer_id IN (:photographerIds)
          AND pc.active = true
          AND (pc.expires_at IS NULL OR pc.expires_at > :now)
          AND (
              pc.usage_limit IS NULL OR pc.usage_limit > (
                  SELECT COUNT(*) FROM orders o
                  WHERE o.coupon_id = pc.id AND o.status <> 'EXPIRED'
              )
          )
        """,
        nativeQuery = true,
    )
    fun findLiveForEvent(
        @Param("eventId") eventId: UUID,
        @Param("photographerIds") photographerIds: Collection<UUID>,
        @Param("now") now: OffsetDateTime,
    ): List<PhotographerCoupon>
}
