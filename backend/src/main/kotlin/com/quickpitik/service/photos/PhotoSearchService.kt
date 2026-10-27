package com.quickpitik.service.photos

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.dto.photos.FaceCandidate
import com.quickpitik.dto.photos.FaceCandidateResult
import com.quickpitik.dto.removal.SearchImage
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.FaceBibProvider
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.contentTypeForKey
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.Timer
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import java.util.UUID
import java.util.concurrent.TimeUnit

@Service
class PhotoSearchService(
    private val aiApiClient: FaceBibProvider,
    private val aiApiProperties: AiApiProperties,
    private val photoService: PhotoService,
    private val userSelfieRepository: UserSelfieRepository,
    private val storageService: StorageService,
    private val meterRegistry: MeterRegistry,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun searchByFace(
        eventId: UUID,
        selfieBytes: ByteArray,
        contentType: String,
        filename: String,
        pagination: PaginationParams,
        requesterUserId: UUID? = null,
        allowFallbackOnError: Boolean = true,
    ): PaginatedResponse<PhotoDto> = timedSearch(MODE_ONESHOT) { trace ->
        faceSearch(eventId, selfieBytes, contentType, filename, pagination, requesterUserId, allowFallbackOnError, trace)
    }

    /**
     * The photos-ready email's check (EventPhotosReadyNotifier): the same
     * one-shot search, timed as mode=alert so background sweeps never skew
     * the runners' numbers, and never answered with the demo fallback grid.
     */
    fun searchByFaceForAlert(
        eventId: UUID,
        selfieBytes: ByteArray,
        contentType: String,
        filename: String,
        pagination: PaginationParams,
        requesterUserId: UUID?,
    ): PaginatedResponse<PhotoDto> = timedSearch(MODE_ALERT) { trace ->
        faceSearch(eventId, selfieBytes, contentType, filename, pagination, requesterUserId, allowFallbackOnError = false, trace)
    }

    private fun faceSearch(
        eventId: UUID,
        selfieBytes: ByteArray,
        contentType: String,
        filename: String,
        pagination: PaginationParams,
        requesterUserId: UUID?,
        allowFallbackOnError: Boolean,
        trace: SearchTrace,
    ): PaginatedResponse<PhotoDto> {
        trace.selfies = 1
        validateSelfie(selfieBytes)
        ensureEnabled()
        val candidates = try {
            timed(null, { trace.aiNanos += it }) {
                findFaceCandidates(eventId, SearchImage(selfieBytes, contentType, filename))
            }
        } catch (ex: Exception) {
            if (allowFallbackOnError && aiApiProperties.searchFallbackOnError && (ex !is AiApiException || ex.isRetryable)) {
                log.warn(
                    "DEMO FALLBACK (app.ai-api.search-fallback-on-error=true): faces/search failed for event {}: {}. " +
                        "Returning the full event grid — these are NOT face matches.",
                    eventId, ex.message,
                )
                return photoService.listForEvent(eventId, null, pagination, requesterUserId)
            }
            throw ex
        }
        val matchedPersonIds = candidates.matches.map { it.personId }.toSet()
        if (matchedPersonIds.isEmpty()) {
            return PaginatedResponse.empty(pagination)
        }
        return timedQuery(trace) {
            photoService.findByEventAndPersonIds(eventId, matchedPersonIds, pagination, requesterUserId)
        }
    }

    fun findFaceCandidates(eventId: UUID, image: SearchImage): FaceCandidateResult {
        validateSelfie(image.bytes)
        ensureEnabled()
        val threshold = aiApiProperties.faceMatchThresholdDefault
        val result = try {
            meterRegistry.timer("qp.ai.call", "op", "search").recordCallable {
                aiApiClient.facesSearch(image.bytes, image.contentType, image.filename, eventId, threshold, CANDIDATE_LIMIT)
            }!!
        } catch (error: Exception) {
            log.warn("ai-api faces/search failed for event {}: {}", eventId, error.message)
            if (error is AiApiException) throw error
            throw AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "Face search is temporarily unavailable", error)
        }
        // The backend applies its threshold even if a provider ignores it.
        val matches = result.matches.filter { it.similarity >= threshold }.map { FaceCandidate(it.person_id, it.similarity) }
        if (matches.size < result.matches.size) {
            log.warn("ai-api returned {} sub-threshold match(es) for event {} (threshold {}); dropped server-side",
                result.matches.size - matches.size, eventId, threshold)
        }
        return FaceCandidateResult(matches, CANDIDATE_LIMIT,
            result.possiblyTruncated ?: (result.matches.size >= CANDIDATE_LIMIT), result.noFaces)
    }

    private fun validateSelfie(bytes: ByteArray) {
        if (bytes.isEmpty()) throw ValidationException(ErrorCodes.SELFIE_REQUIRED, "selfie file is required", "selfie")
    }

    private fun ensureEnabled() {
        if (!aiApiProperties.enabled) throw AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "ai-api is disabled — face search unavailable")
    }

    /** One stored selfie's bytes, as read back from storage. */
    data class SelfieSample(val bytes: ByteArray, val contentType: String, val filename: String)

    /**
     * Search with EVERY selfie in the runner's library (2026-09-02). Each
     * sample is matched on its own and the matched person ids are unioned
     * before the single photo query, so a face caught only from the side or
     * with sunglasses still pulls its photos in and pagination stays exact.
     * One AI call per selfie (library cap is 5). A selfie the provider
     * rejects (no face, too small — a 4xx) is skipped; the search only fails
     * if every selfie was rejected or the provider is down.
     */
    fun searchByFaces(
        eventId: UUID,
        samples: List<SelfieSample>,
        pagination: PaginationParams,
        requesterUserId: UUID? = null,
    ): PaginatedResponse<PhotoDto> = searchByFaces(eventId, samples, pagination, requesterUserId, SearchTrace())

    private fun searchByFaces(
        eventId: UUID,
        samples: List<SelfieSample>,
        pagination: PaginationParams,
        requesterUserId: UUID?,
        trace: SearchTrace,
    ): PaginatedResponse<PhotoDto> {
        trace.selfies = samples.size
        require(samples.isNotEmpty()) {
            throw ValidationException(
                code = ErrorCodes.SELFIE_REQUIRED,
                message = "at least one selfie is required",
                field = "selfieId",
            )
        }
        ensureEnabled()
        val personIds = LinkedHashSet<String>()
        var accepted = 0
        var lastRejection: AiApiException? = null
        for (sample in samples) {
            val matches = try {
                timed(null, { trace.aiNanos += it }) {
                    findFaceCandidates(eventId, SearchImage(sample.bytes, sample.contentType, sample.filename))
                }
            } catch (ex: AiApiException) {
                if (ex.status.is4xxClientError) {
                    lastRejection = ex
                    continue
                }
                throw ex
            } catch (ex: Exception) {
                log.warn("ai-api faces/search failed for event {}: {}", eventId, ex.message)
                throw AiApiException(
                    status = HttpStatus.SERVICE_UNAVAILABLE,
                    aiCode = null,
                    message = "Face search is temporarily unavailable",
                    cause = ex,
                )
            }
            accepted++
            matches.matches.mapTo(personIds) { it.personId }
        }
        if (accepted == 0 && lastRejection != null) throw lastRejection
        if (personIds.isEmpty()) return PaginatedResponse.empty(pagination)
        return timedQuery(trace) { photoService.findByEventAndPersonIds(eventId, personIds, pagination, requesterUserId) }
    }

    /**
     * Stored-selfie face search (moved out of EventPhotoController 2026-09-12,
     * audit P-6). The runner's own selfies are read here so the HTTP edge
     * injects no repository; `findByIdAndUserId` keeps the row scoped to the
     * caller, which is what makes cross-user IDOR impossible.
     *
     * Deliberately NOT transactional: every branch ends in a storage read plus
     * provider HTTP (backend/AGENTS.md rule 4).
     */
    fun searchWithStoredSelfie(
        userId: UUID,
        eventId: UUID,
        selfieId: UUID?,
        allSelfies: Boolean,
        pagination: PaginationParams,
    ): PaginatedResponse<PhotoDto> = timedSearch(if (allSelfies) MODE_LIBRARY else MODE_STORED) { trace ->
        storedSelfieSearch(userId, eventId, selfieId, allSelfies, pagination, trace)
    }

    private fun storedSelfieSearch(
        userId: UUID,
        eventId: UUID,
        selfieId: UUID?,
        allSelfies: Boolean,
        pagination: PaginationParams,
        trace: SearchTrace,
    ): PaginatedResponse<PhotoDto> {
        // allSelfies (2026-09-02): match against the whole library, not one
        // pick. Every selfie the runner saved is a different angle of the same
        // face; unioning them is what lets a side-on race photo match.
        if (allSelfies) {
            val library = userSelfieRepository.findByUserIdOrderByUploadedAtDesc(userId)
            if (library.isEmpty()) {
                throw NotFoundException(code = ErrorCodes.SELFIE_NOT_FOUND, message = "Selfie not found")
            }
            // Skip rows whose object is gone (storage backend changed, deleted
            // out-of-band) rather than failing the whole search on one of them.
            val samples = library.take(MAX_LIBRARY_SELFIES).mapNotNull { s ->
                runCatching { timedFetch(trace) { storageService.getBytes(s.s3Key) } }.getOrNull()?.let { bytes ->
                    SelfieSample(
                        bytes = bytes,
                        contentType = contentTypeForKey(s.s3Key),
                        filename = s.s3Key.substringAfterLast('/'),
                    )
                }
            }
            if (samples.isEmpty()) {
                throw NotFoundException(
                    code = ErrorCodes.SELFIE_NOT_FOUND,
                    message = "Selfie file is no longer available",
                )
            }
            return searchByFaces(eventId, samples, pagination, userId, trace)
        }
        val id = selfieId ?: throw ValidationException(
            code = ErrorCodes.SELFIE_REQUIRED,
            message = "selfieId is required",
            field = "selfieId",
        )
        val selfie = userSelfieRepository.findByIdAndUserId(id, userId)
            ?: throw NotFoundException(code = ErrorCodes.SELFIE_NOT_FOUND, message = "Selfie not found")
        // F6 (2026-05-27): StorageService.getBytes throws if the object is
        // missing from S3 (deleted out-of-band, key never written, etc.).
        // Without this wrap the runner gets a generic 500 INTERNAL_ERROR
        // instead of a meaningful 404 they can react to.
        val bytes = runCatching { timedFetch(trace) { storageService.getBytes(selfie.s3Key) } }.getOrElse {
            throw NotFoundException(
                code = ErrorCodes.SELFIE_NOT_FOUND,
                message = "Selfie file is no longer available",
            )
        }
        return faceSearch(
            eventId = eventId,
            selfieBytes = bytes,
            contentType = contentTypeForKey(selfie.s3Key),
            filename = selfie.s3Key.substringAfterLast('/'),
            pagination = pagination,
            requesterUserId = userId,
            allowFallbackOnError = true,
            trace = trace,
        )
    }

    // Search-path timers (FS-1, 2026-10-27). qp.search.request{mode} times the
    // whole search, failures included; qp.search.selfie.fetch each storage
    // read; qp.search.photo.query the photo lookup. Provider time is already
    // qp.ai.call{op=search}. The INFO line carries the same split per search,
    // so percentiles can be read from the logs without an admin login.

    /** What one face search spent its time on, for its summary line. */
    private class SearchTrace {
        var selfies = 0
        var fetchNanos = 0L
        var aiNanos = 0L
        var queryNanos = 0L
    }

    private fun timedSearch(
        mode: String,
        search: (SearchTrace) -> PaginatedResponse<PhotoDto>,
    ): PaginatedResponse<PhotoDto> {
        val trace = SearchTrace()
        var results = "error"
        return timed(meterRegistry.timer("qp.search.request", "mode", mode), { totalNanos ->
            log.info(
                "face search mode={} selfies={} total_ms={} r2_ms={} aws_ms={} query_ms={} results={}",
                mode, trace.selfies, totalNanos.ms(), trace.fetchNanos.ms(), trace.aiNanos.ms(),
                trace.queryNanos.ms(), results,
            )
        }) { search(trace).also { results = it.total.toString() } }
    }

    private fun <T> timedFetch(trace: SearchTrace, fetch: () -> T): T =
        timed(meterRegistry.timer("qp.search.selfie.fetch"), { trace.fetchNanos += it }, fetch)

    private fun <T> timedQuery(trace: SearchTrace, query: () -> T): T =
        timed(meterRegistry.timer("qp.search.photo.query"), { trace.queryNanos += it }, query)

    /** Runs [block], then records its time (success or failure) on [timer] and hands it to [add]. */
    private inline fun <T> timed(timer: Timer?, add: (Long) -> Unit, block: () -> T): T {
        val start = System.nanoTime()
        try {
            return block()
        } finally {
            val nanos = System.nanoTime() - start
            timer?.record(nanos, TimeUnit.NANOSECONDS)
            add(nanos)
        }
    }

    private fun Long.ms(): Long = TimeUnit.NANOSECONDS.toMillis(this)

    private companion object {
        const val MODE_ONESHOT = "oneshot"
        const val MODE_STORED = "stored"
        const val MODE_LIBRARY = "library"
        const val MODE_ALERT = "alert"
        const val CANDIDATE_LIMIT = 50
        // Mirrors the selfie-library cap (SELFIE_MAX = 5 on both clients).
        const val MAX_LIBRARY_SELFIES = 5
    }
}
