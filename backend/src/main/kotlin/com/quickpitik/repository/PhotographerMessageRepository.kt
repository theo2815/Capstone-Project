package com.quickpitik.repository

import com.quickpitik.entity.PhotographerMessage
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface PhotographerMessageRepository : JpaRepository<PhotographerMessage, UUID> {
    fun findByPhotographerIdOrderByCreatedAtDescIdAsc(photographerId: UUID): List<PhotographerMessage>
}
