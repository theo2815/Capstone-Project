package com.quickpitik.repository

import com.quickpitik.entity.SocialLink
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface SocialLinkRepository : JpaRepository<SocialLink, UUID> {
    fun findAllByUserIdOrderByCreatedAtAsc(userId: UUID): List<SocialLink>
    fun findByIdAndUserId(id: UUID, userId: UUID): SocialLink?
    fun countByUserId(userId: UUID): Long
}
