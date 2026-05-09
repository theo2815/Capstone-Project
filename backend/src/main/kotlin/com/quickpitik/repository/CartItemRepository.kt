package com.quickpitik.repository

import com.quickpitik.entity.CartItemEntity
import com.quickpitik.entity.CartItemId
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

interface CartItemRepository : JpaRepository<CartItemEntity, CartItemId> {

    @Query("SELECT c FROM CartItemEntity c WHERE c.id.userId = :userId ORDER BY c.addedAt DESC")
    fun findByUserId(@Param("userId") userId: UUID): List<CartItemEntity>

    @Modifying
    @Transactional
    @Query("DELETE FROM CartItemEntity c WHERE c.id.userId = :userId AND c.id.photoId = :photoId")
    fun deleteByUserIdAndPhotoId(@Param("userId") userId: UUID, @Param("photoId") photoId: UUID): Int

    @Modifying
    @Transactional
    @Query("DELETE FROM CartItemEntity c WHERE c.id.userId = :userId")
    fun deleteAllByUserId(@Param("userId") userId: UUID): Int
}
