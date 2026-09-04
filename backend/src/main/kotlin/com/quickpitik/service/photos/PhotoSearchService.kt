package com.quickpitik.service.photos

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.FaceBibProvider
import io.micrometer.core.instrument.MeterRegistry
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import java.util.UUID

@Service
class PhotoSearchService(
    private val aiApiClient: FaceBibProvider,
    private val aiApiProperties: AiApiProperties,
    private val photoService: PhotoService,
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
    ): PaginatedResponse<PhotoDto> {
        require(selfieBytes.isNotEmpty()) {
            throw ValidationException(
                code = ErrorCodes.SELFIE_REQUIRED,
                message = "selfie file is required",
                field = "selfie",
            )
        }
        if (!aiApiProperties.enabled) {
            // Feature-dev short-circuit. ai-api is intentionally off; surface
            // the same 503 envelope a real outage would. Bib search keeps
            // working because it's a DB query (just empty until AI is on
            // during upload to populate photo_bibs).
            throw AiApiException(
                status = HttpStatus.SERVICE_UNAVAILABLE,
                aiCode = null,
                message = "ai-api is disabled — face search unavailable",
            )
        }
        val threshold = aiApiProperties.faceMatchThresholdDefault
        val matches = try {
            meterRegistry.timer("qp.ai.call", "op", "search").recordCallable {
                aiApiClient.facesSearch(
                    file = selfieBytes,
                    contentType = contentType,
                    filename = filename,
                    eventId = eventId,
                    threshold = threshold,
                    topK = 50,
                )
            }!!
        } catch (ex: Exception) {
            if (allowFallbackOnError && aiApiProperties.searchFallbackOnError) {
                log.warn(
                    "DEMO FALLBACK (app.ai-api.search-fallback-on-error=true): faces/search failed for event {}: {}. " +
                        "Returning the full event grid — these are NOT face matches.",
                    eventId, ex.message,
                )
                return photoService.listForEvent(eventId, null, pagination, requesterUserId)
            }
            log.warn("ai-api faces/search failed for event {}: {}", eventId, ex.message)
            // Rethrow AiApiException untouched so GlobalExceptionHandler can map
            // its real status (a 4xx from ai-api is a bad selfie, not an outage).
            // Anything else is a genuine failure to reach ai-api → 503.
            if (ex is AiApiException) throw ex
            throw AiApiException(
                status = HttpStatus.SERVICE_UNAVAILABLE,
                aiCode = null,
                message = "Face search is temporarily unavailable",
                cause = ex,
            )
        }

        // Defense in depth: ai-api is told the threshold, but we don't rely on
        // it having honoured it. A sub-threshold match here means someone else's
        // photos would surface as "yours" — the one failure mode face search
        // must never have.
        val qualifying = matches.matches.filter { it.similarity >= threshold }
        val dropped = matches.matches.size - qualifying.size
        if (dropped > 0) {
            log.warn(
                "ai-api returned {} sub-threshold match(es) for event {} (threshold {}); dropped server-side",
                dropped, eventId, threshold,
            )
        }

        val matchedPersonIds = qualifying.map { it.person_id }.toSet()
        if (matchedPersonIds.isEmpty()) {
            return PaginatedResponse.empty(pagination)
        }
        return photoService.findByEventAndPersonIds(eventId, matchedPersonIds, pagination, requesterUserId)
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
    ): PaginatedResponse<PhotoDto> {
        require(samples.isNotEmpty()) {
            throw ValidationException(
                code = ErrorCodes.SELFIE_REQUIRED,
                message = "at least one selfie is required",
                field = "selfieId",
            )
        }
        if (!aiApiProperties.enabled) {
            throw AiApiException(
                status = HttpStatus.SERVICE_UNAVAILABLE,
                aiCode = null,
                message = "ai-api is disabled — face search unavailable",
            )
        }
        val threshold = aiApiProperties.faceMatchThresholdDefault
        val personIds = LinkedHashSet<String>()
        var accepted = 0
        var lastRejection: AiApiException? = null
        for (sample in samples) {
            val matches = try {
                meterRegistry.timer("qp.ai.call", "op", "search").recordCallable {
                    aiApiClient.facesSearch(
                        file = sample.bytes,
                        contentType = sample.contentType,
                        filename = sample.filename,
                        eventId = eventId,
                        threshold = threshold,
                        topK = 50,
                    )
                }!!
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
            matches.matches.filter { it.similarity >= threshold }.mapTo(personIds) { it.person_id }
        }
        if (accepted == 0 && lastRejection != null) throw lastRejection
        if (personIds.isEmpty()) return PaginatedResponse.empty(pagination)
        return photoService.findByEventAndPersonIds(eventId, personIds, pagination, requesterUserId)
    }
}
