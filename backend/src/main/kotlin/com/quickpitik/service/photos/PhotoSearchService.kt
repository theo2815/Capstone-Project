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
        val matches = try {
            aiApiClient.facesSearch(
                file = selfieBytes,
                contentType = contentType,
                filename = filename,
                eventId = eventId,
                threshold = aiApiProperties.faceMatchThresholdDefault,
                topK = 50,
            )
        } catch (ex: AiApiException) {
            log.warn("ai-api faces/search failed for event {}: {}", eventId, ex.message)
            if (ex.aiCode == "LOW_QUALITY" || ex.aiCode == "NO_FACES") {
                throw ValidationException(
                    code = ErrorCodes.SELFIE_REJECTED,
                    message = ex.message ?: "Selfie rejected",
                    field = "selfie",
                )
            }
            throw ex
        }

        val matchedPersonIds = matches.matches.map { it.person_id }.toSet()
        if (matchedPersonIds.isEmpty()) {
            return PaginatedResponse.empty(pagination)
        }
        return photoService.findByEventAndPersonIds(eventId, matchedPersonIds, pagination)
    }
}
