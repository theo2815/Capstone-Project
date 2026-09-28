package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.EventMetadataVisibilityDto
import com.quickpitik.dto.photographer.MetadataSettingsDto
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

@Service
@Transactional
class PhotoMetadataSettingsService(
    private val settings: PhotographerSettingsRepository,
    private val photographerSettingsService: PhotographerSettingsService,
    private val coverage: EventPhotographerRepository,
    private val events: EventRepository,
) {
    @Transactional(readOnly = true)
    fun getDefault(userId: UUID): MetadataSettingsDto =
        MetadataSettingsDto(settings.findById(userId).orElse(null)?.metadataPublicByDefault ?: false)

    fun setDefault(userId: UUID, value: Boolean): MetadataSettingsDto {
        photographerSettingsService.getOrCreate(userId).metadataPublicByDefault = value
        return MetadataSettingsDto(value)
    }

    @Transactional(readOnly = true)
    fun getEvent(userId: UUID, eventId: UUID): EventMetadataVisibilityDto {
        val saved = coverage.findById(EventPhotographerId(eventId, userId)).orElse(null)
        if (saved != null) return EventMetadataVisibilityDto(saved.metadataPublic)
        ownedEventOrThrow(userId, eventId)
        return EventMetadataVisibilityDto(false)
    }

    fun setEvent(userId: UUID, eventId: UUID, value: Boolean): EventMetadataVisibilityDto {
        if (coverage.updateMetadataVisibility(eventId, userId, value) == 0) {
            ownedEventOrThrow(userId, eventId)
            coverage.upsertMetadataVisibility(eventId, userId, value)
        }
        return EventMetadataVisibilityDto(value)
    }

    private fun ownedEventOrThrow(userId: UUID, eventId: UUID) {
        if (events.findByIdAndCreatedByAndDeletedAtIsNull(eventId, userId) == null) {
            throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        }
    }
}
