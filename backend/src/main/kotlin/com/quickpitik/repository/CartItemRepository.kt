package com.quickpitik.repository

import com.quickpitik.entity.CartItemEntity
import com.quickpitik.entity.CartItemId
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.util.UUID

interface CartItemRepository : JpaRepository<CartItemEntity, CartItemId> {

    @Query("SELECT c FROM CartItemEntity c WHERE c.id.userId = :userId ORDER BY c.addedAt DESC")
    fun findByUserId(@Param("userId") userId: UUID): List<CartItemEntity>

    @Modifying
    @Transactional
    @Query("DELETE FROM CartItemEntity c WHERE c.id.userId = :userId AND c.id.photoId = :photoId")
    fun deleteByUserIdAndPhotoId(@Param("userId") userId: UUID, @Param("photoId") photoId: UUID): Int

    /** AF-12: moves a stored price up to the live one; never down, never another buyer's row. */
    @Modifying
    @Transactional
    @Query(
        "UPDATE CartItemEntity c SET c.pricePhpAtAdd = :live " +
            "WHERE c.id.userId = :userId AND c.id.photoId = :photoId AND c.pricePhpAtAdd < :live",
    )
    fun raisePriceAtAdd(
        @Param("userId") userId: UUID,
        @Param("photoId") photoId: UUID,
        @Param("live") live: BigDecimal,
    ): Int

    /** AF-18 F5: moves a stored price down to a lower live price the cart answer shows; never up. */
    @Modifying
    @Transactional
    @Query(
        "UPDATE CartItemEntity c SET c.pricePhpAtAdd = :live " +
            "WHERE c.id.userId = :userId AND c.id.photoId = :photoId AND c.pricePhpAtAdd > :live",
    )
    fun lowerPriceAtAdd(
        @Param("userId") userId: UUID,
        @Param("photoId") photoId: UUID,
        @Param("live") live: BigDecimal,
    ): Int

    @Modifying
    @Transactional
    @Query("DELETE FROM CartItemEntity c WHERE c.id.userId = :userId")
    fun deleteAllByUserId(@Param("userId") userId: UUID): Int
}
