package com.quickpitik.service.ai

import com.quickpitik.dto.ai.BibsRecognizeResult
import com.quickpitik.dto.ai.FacesDetectResult
import com.quickpitik.dto.ai.FacesEnrollResult
import com.quickpitik.dto.ai.FacesSearchResult
import java.util.UUID

// The synchronous face + bib inference the hot upload/search paths depend on,
// abstracted over the provider that fulfils it. Two implementations:
//   - AiApiClient          — the self-hosted ai-api service (default)
//   - RekognitionAiClient  — AWS Rekognition (app.ai.provider=rekognition)
// The ai-api-only async surface (mega jobs, webhooks, jobStatus) stays on
// AiApiClient — it has no Rekognition analogue and is inert under that provider.
interface FaceBibProvider {
    fun facesDetect(file: ByteArray, contentType: String, filename: String): FacesDetectResult

    fun facesEnroll(
        file: ByteArray,
        contentType: String,
        filename: String,
        personName: String,
        personId: String?,
        eventId: UUID,
    ): FacesEnrollResult

    fun facesSearch(
        file: ByteArray,
        contentType: String,
        filename: String,
        eventId: UUID,
        threshold: Double,
        topK: Int,
    ): FacesSearchResult

    fun bibsRecognize(
        file: ByteArray,
        contentType: String,
        filename: String,
        minChars: Int? = null,
    ): BibsRecognizeResult

    fun deleteFacesPerson(personId: String)

    fun deleteFacesByEvent(eventId: UUID)

    fun listPersonsForEvent(eventId: UUID): List<AiPersonRef>
}
