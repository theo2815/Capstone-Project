package com.quickpitik.service.events

import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.Event
import com.quickpitik.repository.EventRepository
import com.quickpitik.service.ai.FaceBibProvider
import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.LocalDate
import java.time.ZoneId
import java.util.UUID

// Legal review M2 (2026-11-04): face templates may be kept only as long as
// their purpose needs (RA 10173 Sec. 11(e)). An event's templates (its
// Rekognition collection) are deleted `faceRetention` (6 months) after the
// event date, or when the event is deleted; from then on face search on the
// event answers 410 FACE_SEARCH_ENDED and its photos are no longer face-indexed.
// Bib search is untouched. Single-instance daily sweep like the other sweeps; a
// failed delete leaves faces_deleted_at empty, so the next run retries.
@Component
class EventFaceExpiry(
    private val eventRepository: EventRepository,
    private val faceBibProvider: FaceBibProvider,
    private val aiApiProperties: AiApiProperties,
    private val clock: Clock,
    private val transactionTemplate: TransactionTemplate,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun ended(event: Event): Boolean =
        event.facesDeletedAt != null || !event.date.plus(aiApiProperties.faceRetention).isAfter(today())

    fun ended(eventId: UUID): Boolean = eventRepository.findById(eventId).map(::ended).orElse(false)

    /** Deletes the event's face templates, then stamps faces_deleted_at. Throws if the delete failed. */
    fun deleteFaces(eventId: UUID) {
        faceBibProvider.deleteFacesByEvent(eventId)
        transactionTemplate.execute {
            eventRepository.findById(eventId).ifPresent {
                it.facesDeletedAt = clock.postgresTimestamp()
                eventRepository.save(it)
            }
        }
    }

    @Scheduled(fixedDelayString = "\${app.ai-api.face-expiry-sweep-interval-ms:86400000}")
    fun sweep() {
        if (!aiApiProperties.enabled) return // same switch AdminEventService.delete uses
        // shortcut: loads every due event in one list; page it if one run ever has thousands.
        val due = (eventRepository.findByFacesDeletedAtIsNullAndDateLessThanEqual(today().minus(aiApiProperties.faceRetention)) +
            eventRepository.findByFacesDeletedAtIsNullAndDeletedAtIsNotNull()).distinctBy { it.id }
        if (due.isEmpty()) return
        log.info("Face expiry sweep: deleting face templates of {} event(s)", due.size)
        due.forEach { event ->
            try {
                deleteFaces(event.id)
            } catch (ex: Exception) {
                log.warn("Face template delete failed for event {} (retried next run): {}", event.id, ex.message)
            }
        }
    }

    private fun today(): LocalDate = LocalDate.now(clock.withZone(PH_ZONE))

    private companion object {
        val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")
    }
}
