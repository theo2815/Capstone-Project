package com.quickpitik.repository

import com.quickpitik.entity.ProcessedWebhookEvent
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface ProcessedWebhookEventRepository : JpaRepository<ProcessedWebhookEvent, UUID> {
    fun existsByProviderAndEventId(provider: String, eventId: String): Boolean
}
