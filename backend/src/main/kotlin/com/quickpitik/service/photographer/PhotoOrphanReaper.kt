package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.photos.PhotoFaceErasureService
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import java.time.OffsetDateTime
import java.util.UUID

// Sweeps ai-api for orphaned person rows — a person enrolled for one of this
// backend's photos that no live photo references anymore. Orphans arise when:
//   - a photo is re-indexed (reconcile retry): index() enrolls a NEW person and
//     the backend keeps only that id, abandoning the prior person on ai-api;
//   - a batch's mega job created persons the backend never ingested (the job
//     went NOT_FOUND / was rolled back, then re-drained into a fresh set).
// Left alone they bloat the pgvector index and leave biometric data with no
// owner record. This is housekeeping, not a hot path — it runs slowly.
//
// Safety: require photo attribution, no unfinished enrollment, no live person
// reference and age beyond REAP_GRACE_MINUTES. The age gate spares a
// person an in-flight index() just enrolled is unreferenced only until its
// facePersons row commits (sub-second), and its fresh created_at keeps it out
// of the orphan set; age never proves an outstanding writer has ended.
@Component
class PhotoOrphanReaper(
    private val photoRepository: PhotoRepository,
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
    private val faceErasure: PhotoFaceErasureService,
    private val removalProperties: com.quickpitik.config.PhotoRemovalProperties,
    private val clock: java.time.Clock,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(fixedDelayString = "\${app.ai-api.reaper-interval-ms:1800000}")
    fun reap() {
        if (!aiApiProperties.enabled || !aiApiProperties.reaperEnabled) return
        // Durable targets survive the last photo/event catalog reference.
        photoRepository.faceWorkPending(PageRequest.of(0, REAP_MAX_EVENTS_PER_SWEEP)).forEach {
            // Persist the backoff before I/O so crashes and persistent failures
            // do not monopolize the bounded first page after restart.
            photoRepository.deferFaceCleanup(it.eventId, it.photoId, OffsetDateTime.now(clock).plus(removalProperties.cleanupBackoff))
            runCatching { faceErasure.erase(it.eventId, it.photoId) }
                .onFailure { error -> log.warn("Face cleanup pending for photo {} ({})", it.photoId, error.javaClass.simpleName) }
        }
        val events = photoRepository.findEventsWithFacePersons(
            PageRequest.of(0, REAP_MAX_EVENTS_PER_SWEEP),
        )
        if (events.isEmpty()) return
        val cutoff = OffsetDateTime.now().minusMinutes(REAP_GRACE_MINUTES)
        var reaped = 0
        events.forEach { eventId ->
            try {
                reaped += reapEvent(eventId, cutoff)
            } catch (ex: Exception) {
                // One event's failure (ai-api blip, etc.) must not stop the rest;
                // the next sweep retries it.
                log.warn("Orphan reap failed for event {}: {}", eventId, ex.message)
            }
        }
        if (reaped > 0) {
            log.info("Reaped {} orphan ai-api person(s) across {} event(s)", reaped, events.size)
        }
    }

    // One event: delete every ai-api person not referenced by a live photo and
    // older than the grace window. Each delete is best-effort — a person already
    // gone (concurrent delete → 404) is a no-op for our purposes.
    private fun reapEvent(eventId: UUID, cutoff: OffsetDateTime): Int {
        val persons = aiApiClient.listPersonsForEvent(eventId)
        if (persons.isEmpty()) return 0
        val referenced = photoRepository.findReferencedAiPersonIds(eventId).toHashSet()
        val orphans = persons.filter { it.id !in referenced && it.createdAt.isBefore(cutoff) }
        var deleted = 0
        orphans.forEach { person ->
            // Names are backend-generated photo UUIDs. An unattributed legacy
            // person is an exception, never an arbitrary deletion target.
            val photoId = runCatching { UUID.fromString(person.personName) }.getOrNull()
            if (photoId == null) {
                log.warn("Orphan face attribution unresolved for person {}", person.id)
                return@forEach
            }
            if (photoRepository.hasUnfinishedWork(photoId, "INDEXING")) return@forEach
            runCatching { aiApiClient.erasePhotoFaces(eventId, person.id) }
                .onSuccess { deleted++ }
                .onFailure { log.warn("Failed to delete orphan person {}: {}", person.id, it.message) }
        }
        return deleted
    }

    private companion object {
        // Only reap persons enrolled longer ago than this — guards the brief
        // window between an index() enrolling a person and its facePersons row
        // committing, so a freshly-created person is never seen as an orphan.
        const val REAP_GRACE_MINUTES = 30L
        // Distinct indexed events checked per sweep. At demo scale this covers
        // every event; any beyond it are picked up on the next pass.
        const val REAP_MAX_EVENTS_PER_SWEEP = 50
    }
}
