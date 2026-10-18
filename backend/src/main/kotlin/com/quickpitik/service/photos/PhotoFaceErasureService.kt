package com.quickpitik.service.photos

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.ai.FaceErasureResult
import com.quickpitik.entity.FaceProvider
import com.quickpitik.entity.Photo
import com.quickpitik.repository.PhotoFaceWork
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.ai.faceResultPhoto
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID

/** Face-only reconciliation. TASK 10 owns authorization, blobs and erasure jobs. */
@Service
class PhotoFaceErasureService(
    private val photos: PhotoRepository,
    private val ai: AiApiClient,
    private val rekognition: ObjectProvider<RekognitionAiClient>,
    private val props: AiApiProperties,
    private val fence: PhotoWorkFence,
    private val json: ObjectMapper,
    private val tx: TransactionTemplate,
    private val clock: Clock,
) {
    fun erase(eventId: UUID, photoId: UUID): FaceErasureResult = guarded {
        val photo = photos.findById(photoId).orElse(null)
        if (photo != null && photo.eventId != eventId) unavailable("Face target event mismatch")
        val work = photos.faceWorkForPhoto(eventId, photoId)
        if (work.isEmpty() && photo == null) unavailable("Face attribution missing")
        val neverIndexed = photo != null && photo.facePersons.isEmpty() && photo.indexingAttempts == 0 &&
            photo.indexedProvider == null && work.isEmpty() && !photos.hasUnfinishedWork(photoId, "INDEXING")
        if (neverIndexed) return@guarded FaceErasureResult.ALREADY_ABSENT
        if (!props.enabled) unavailable("Face cleanup disabled")
        val legacyNoFaces = work.isEmpty() && photo?.facePersons?.isEmpty() == true
        // Every Rekognition face of a photo carries ExternalImageId "<event>.<photo>", so listing it proves absence.
        if (legacyNoFaces && photo?.indexedProvider != "rekognition") unavailable("Legacy face attribution unresolved")

        val targets = mutableListOf<Pair<FaceProvider, String>>()
        if (legacyNoFaces) targets.add(FaceProvider.REKOGNITION to "$eventId.$photoId")
        photo?.facePersons?.forEach { targets.add(it.provider to it.aiPersonId) }
        val expectedIds = mutableMapOf<UUID, String>()
        var unresolved = false
        work.forEach { output ->
            if (output.personName != photoId.toString()) unavailable("Face attribution missing")
            val provider = provider(output.provider)
            val ids = strings(output.personIds).toMutableList()
            // Legacy ids resolve by their shape in eraseTargets; an unknown row without ids proves nothing.
            if (provider == FaceProvider.UNKNOWN && ids.isEmpty()) unavailable("Recorded face provider unresolved")
            targets.addAll(strings(output.priorPersonIds).map { provider(output.priorProvider) to it })
            val claim = claim(output)
            if (output.finishedAt == null) {
                val jobs = json.readTree(output.jobIds)
                val faceJob = jobs.path("FACE").takeIf { it.isTextual }?.asText()
                val bibJob = jobs.path("BIB").takeIf { it.isTextual }?.asText()
                var ended = false
                if (provider == FaceProvider.AI_API && faceJob != null && bibJob != null) {
                    val face = ai.jobStatusComplete(faceJob)
                    val bib = ai.jobStatusComplete(bibJob)
                    if (face.status == "completed" && bib.status in setOf("completed", "failed")) {
                        val batchIds = strings(output.batchPhotoIds).map(UUID::fromString)
                        val results = face.result ?: unavailable("Incomplete terminal job results")
                        val attributed = results.mapIndexed { index, item ->
                            val map = item as? Map<*, *> ?: unavailable("Invalid job attribution")
                            val id = faceResultPhoto(map, index, batchIds) ?: unavailable("Invalid job attribution or outcome")
                            id to map
                        }
                        if (batchIds.isEmpty() || attributed.size != batchIds.size || attributed.map { it.first }.toSet() != batchIds.toSet()) {
                            unavailable("Incomplete terminal job results")
                        }
                        attributed.filter { it.first == photoId }.forEach { (_, map) ->
                            val id = map["person_id"] as? String
                            if (!id.isNullOrBlank() && id !in ids) {
                                fence.recordPerson(claim, id)
                                ids.add(id)
                            }
                        }
                        fence.finish(claim)
                        ended = true
                    }
                }
                if (!ended) unresolved = true
            }
            // A lost response can produce more than one person for this name.
            // Capture every observed target, even when its writer is unresolved.
            if (provider == FaceProvider.AI_API) {
                ai.listPersonsForEvent(eventId).filter { it.personName == photoId.toString() }.forEach {
                    if (it.id !in ids) {
                        fence.recordPerson(claim, it.id)
                        ids.add(it.id)
                    }
                }
            }
            targets.addAll(ids.map { provider to it })
            if (provider == FaceProvider.REKOGNITION) targets.add(provider to "$eventId.$photoId")
            expectedIds[output.token] = json.writeValueAsString(ids)
        }
        if (unresolved || photos.hasUnfinishedWork(photoId, "INDEXING")) unavailable("Face enrollment unresolved")
        val result = eraseTargets(eventId, photoId, targets)
        tx.executeWithoutResult {
            photos.findAllByIdForUpdate(listOf(photoId))
            if (photos.hasUnfinishedWork(photoId, "INDEXING")) unavailable("Face enrollment unresolved")
            val current = photos.faceWorkForPhoto(eventId, photoId)
            if (current.map { it.token }.toSet() != work.map { it.token }.toSet()) {
                unavailable("Face targets changed during cleanup")
            }
            work.forEach {
                if (photos.confirmFaceWorkErased(it.token, expectedIds.getValue(it.token), OffsetDateTime.now(clock)) != 1) {
                    unavailable("Face targets changed during cleanup")
                }
            }
        }
        result
    }

    // Called only after the producer has saved its claim/prior targets and
    // before it submits new enrollment. Never settles that new writer's token.
    fun erasePrior(photo: Photo) = guarded {
        if (photo.facePersons.isNotEmpty()) {
            eraseTargets(photo.eventId, photo.id, photo.facePersons.map { it.provider to it.aiPersonId })
        }
        Unit
    }

    private fun eraseTargets(eventId: UUID, photoId: UUID, original: List<Pair<FaceProvider, String>>): FaceErasureResult {
        if (original.isEmpty()) return FaceErasureResult.ALREADY_ABSENT
        if (!props.enabled) unavailable("Face cleanup disabled")
        val aiNeeded = original.any { (p, id) -> p == FaceProvider.AI_API || (p == FaceProvider.UNKNOWN && !id.contains('.')) }
        val listing = if (aiNeeded) ai.listPersonsForEvent(eventId) else emptyList()
        val resolved = original.map { (p, id) ->
            val actual = if (p != FaceProvider.UNKNOWN) p else if (id == "$eventId.$photoId") FaceProvider.REKOGNITION
            else if (listing.any { it.id == id && it.personName == photoId.toString() }) FaceProvider.AI_API
            else unavailable("Legacy face attribution unresolved")
            if (actual == FaceProvider.REKOGNITION && id != "$eventId.$photoId") unavailable("Face target event mismatch")
            if (actual == FaceProvider.AI_API && listing.any { it.id == id && it.personName != photoId.toString() }) {
                unavailable("Face target photo mismatch")
            }
            actual to id
        }.toMutableSet()
        rememberAttribution(eventId, photoId, resolved.associate { it.second to it.first },
            original.filter { it.first == FaceProvider.UNKNOWN }.map { it.second }.toSet())
        if (resolved.any { it.first == FaceProvider.AI_API }) {
            resolved.addAll(listing.filter { it.personName == photoId.toString() }.map { FaceProvider.AI_API to it.id })
        }
        var deleted = false
        resolved.forEach { (provider, id) ->
            val result = when (provider) {
                FaceProvider.AI_API -> ai.erasePhotoFaces(eventId, id)
                FaceProvider.REKOGNITION -> (rekognition.ifAvailable ?: unavailable("Recorded face provider unavailable")).erasePhotoFaces(eventId, id)
                FaceProvider.UNKNOWN -> unavailable("Legacy face attribution unresolved")
            }
            if (result == FaceErasureResult.DELETED) deleted = true
        }
        return if (deleted) FaceErasureResult.DELETED else FaceErasureResult.ALREADY_ABSENT
    }

    private fun rememberAttribution(eventId: UUID, photoId: UUID, proven: Map<String, FaceProvider>, required: Set<String>) {
        tx.executeWithoutResult {
            val durable = mutableSetOf<String>()
            val photo = photos.findAllByIdForUpdate(listOf(photoId)).singleOrNull()
            if (photo != null) {
                if (photo.eventId != eventId) unavailable("Face target event mismatch")
                var changed = false
                photo.facePersons.forEach { face ->
                    val actual = proven[face.aiPersonId] ?: return@forEach
                    if (face.provider == FaceProvider.UNKNOWN) { face.provider = actual; changed = true }
                    else if (face.provider != actual) unavailable("Face provenance changed during cleanup")
                    durable.add(face.aiPersonId)
                }
                if (changed) photos.saveAndFlush(photo)
            }
            photos.faceWorkForPhoto(eventId, photoId).forEach { output ->
                listOf(false, true).forEach { prior ->
                    val rawIds = if (prior) output.priorPersonIds else output.personIds
                    val ids = strings(rawIds)
                    val stored = provider(if (prior) output.priorProvider else output.provider)
                    if (ids.any { it in proven } && output.personName != photoId.toString()) unavailable("Face attribution missing")
                    if (stored == FaceProvider.UNKNOWN && ids.any { it in proven }) {
                        val actual = ids.map { proven[it] ?: unavailable("Legacy face attribution unresolved") }.toSet()
                        if (actual.size != 1) unavailable("Legacy face attribution ambiguous")
                        if (photos.attributeFaceWorkProvider(output.token, eventId, photoId, prior, rawIds, actual.single().name.lowercase()) != 1) {
                            unavailable("Face targets changed during cleanup")
                        }
                        durable.addAll(ids)
                    } else if (stored != FaceProvider.UNKNOWN) {
                        durable.addAll(ids.filter { proven[it] == stored })
                    }
                }
            }
            if (!durable.containsAll(required)) unavailable("Face targets changed during cleanup")
        }
    }

    private fun strings(raw: String): List<String> {
        val node = json.readTree(raw)
        if (!node.isArray || node.any { !it.isTextual || it.asText().isBlank() }) unavailable("Invalid face provenance")
        return node.map { it.asText() }
    }

    private fun provider(raw: String?): FaceProvider =
        raw?.uppercase()?.let { runCatching { FaceProvider.valueOf(it) }.getOrNull() } ?: FaceProvider.UNKNOWN

    private fun claim(work: PhotoFaceWork) = PhotoWorkClaim(work.photoId, work.generation, work.token, work.expiresAt, PhotoWorkKind.INDEXING)

    private fun unavailable(reason: String): Nothing =
        throw AiApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.AI_API_UNAVAILABLE, reason)

    private fun <T> guarded(block: () -> T): T = try {
        block()
    } catch (ex: AiApiException) {
        throw ex
    } catch (ex: Exception) {
        unavailable("Face cleanup provider unavailable")
    }
}
