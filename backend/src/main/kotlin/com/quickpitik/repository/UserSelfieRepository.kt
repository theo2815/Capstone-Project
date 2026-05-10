package com.quickpitik.repository

import com.quickpitik.entity.UserSelfie
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.util.UUID

@Repository
interface UserSelfieRepository : JpaRepository<UserSelfie, UUID> {

    fun findByUserIdOrderByUploadedAtDesc(userId: UUID): List<UserSelfie>

    fun countByUserId(userId: UUID): Long

    fun findByIdAndUserId(id: UUID, userId: UUID): UserSelfie?

    fun findFirstByUserIdAndIsPrimaryTrue(userId: UUID): UserSelfie?

    fun findFirstByUserIdOrderByUploadedAtDesc(userId: UUID): UserSelfie?

    @Modifying
    @Query(
        """
        UPDATE UserSelfie us
        SET us.isPrimary = false
        WHERE us.userId = :userId AND us.isPrimary = true AND us.id <> :keepId
        """,
    )
    fun demoteOtherPrimaries(@Param("userId") userId: UUID, @Param("keepId") keepId: UUID): Int
}
