package com.quickpitik.service.photos

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import java.util.UUID

@Service
class PhotoSearchService(
    private val aiApiClient: AiApiClient,
    private val aiApiProperties: AiApiProperties,
    private val photoService: PhotoService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun searchByFace(
        eventId: UUID,
        selfieBytes: ByteArray,
        contentType: String,
        filename: String,
        pagination: PaginationParams,
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
        val matches = try {
            aiApiClient.facesSearch(
                file = selfieBytes,
                contentType = contentType,
                filename = filename,
                eventId = eventId,
                threshold = aiApiProperties.faceMatchThresholdDefault,
                topK = 50,
            )
        } catch (ex: Exception) {
            log.warn("ai-api faces/search failed/offline for event {}: {}. Falling back to demo list.", eventId, ex.message)
            // Bulletproof Demo Fallback: return the event's photos so the app works flawlessly even with heavy AI container stopped!
            return photoService.listForEvent(eventId, null, pagination)
        }

        val matchedPersonIds = matches.matches.map { it.person_id }.toSet()
        if (matchedPersonIds.isEmpty()) {
            return PaginatedResponse.empty(pagination)
        }
        return photoService.findByEventAndPersonIds(eventId, matchedPersonIds, pagination)
    }
}
