package com.quickpitik.repository

import com.quickpitik.entity.SavedEvent
import com.quickpitik.entity.SavedEventId
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

interface SavedEventRepository : JpaRepository<SavedEvent, SavedEventId> {

    @Query("SELECT s.id.eventId FROM SavedEvent s WHERE s.id.userId = :userId ORDER BY s.savedAt DESC")
    fun findEventIdsByUserId(@Param("userId") userId: UUID): List<UUID>

    // Returns SavedEvent rows in saved-at-desc order so the service can zip
    // with Event lookups to build SavedEventSummaryDto without losing ordering.
    @Query("SELECT s FROM SavedEvent s WHERE s.id.userId = :userId ORDER BY s.savedAt DESC")
    fun findByUserIdOrderedBySavedAtDesc(@Param("userId") userId: UUID): List<SavedEvent>

    @Modifying
    @Transactional
    @Query("DELETE FROM SavedEvent s WHERE s.id.userId = :userId AND s.id.eventId = :eventId")
    fun deleteByUserIdAndEventId(@Param("userId") userId: UUID, @Param("eventId") eventId: UUID): Int
}
