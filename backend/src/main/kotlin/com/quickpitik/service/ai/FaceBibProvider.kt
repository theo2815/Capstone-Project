package com.quickpitik.service.ai

import com.quickpitik.dto.ai.BibsRecognizeResult
import com.quickpitik.dto.ai.FacesDetectResult
import com.quickpitik.dto.ai.FacesEnrollResult
import com.quickpitik.dto.ai.FacesSearchResult
import com.quickpitik.dto.ai.FaceErasureResult
import java.util.UUID

// The synchronous face + bib inference the hot upload/search paths depend on,
// abstracted over the provider that fulfils it. Two implementations:
//   - RekognitionAiClient  — AWS Rekognition (app.ai.provider=rekognition, default)
//   - AiApiClient          — the self-hosted ai-api service, which no longer has
//                            face/bib routes (removed 2026-10-16)
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

    fun erasePhotoFaces(eventId: UUID, personId: String): FaceErasureResult

    fun deleteFacesByEvent(eventId: UUID)

    fun listPersonsForEvent(eventId: UUID): List<AiPersonRef>
}
