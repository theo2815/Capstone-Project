package com.quickpitik.service.ai

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.ai.AiApiEnvelope
import com.quickpitik.dto.ai.BibsRecognizeResult
import com.quickpitik.dto.ai.FacesDetectResult
import com.quickpitik.dto.ai.FacesEnrollResult
import com.quickpitik.dto.ai.FacesSearchResult
import com.quickpitik.dto.ai.HealthReady
import com.quickpitik.dto.ai.FaceErasureResult
import com.quickpitik.common.ErrorCodes
import com.fasterxml.jackson.databind.JsonNode
import com.quickpitik.dto.ai.JobCreateResult
import com.quickpitik.dto.ai.JobStatusResult
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.core.io.ByteArrayResource
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.stereotype.Service
import org.springframework.util.LinkedMultiValueMap
import org.springframework.util.MultiValueMap
import org.springframework.web.client.HttpClientErrorException
import org.springframework.web.client.HttpServerErrorException
import org.springframework.web.client.ResourceAccessException
import org.springframework.web.client.RestClient
import java.time.OffsetDateTime
import java.util.UUID

// One enrolled person as the reaper needs it: its id plus when ai-api created
// it (used to spare just-enrolled persons from being mistaken for orphans).
data class AiPersonRef(val id: String, val createdAt: OffsetDateTime, val personName: String? = null)

@Service
class AiApiClient(
    @Qualifier("aiApiRestClient") private val client: RestClient,
    private val props: AiApiProperties,
    private val objectMapper: ObjectMapper,
) : FaceBibProvider {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun facesDetect(file: ByteArray, contentType: String, filename: String): FacesDetectResult {
        val body = singleFileBody(file, contentType, filename)
        return postAndUnwrap("/api/v1/faces/detect", body, facesDetectRef)
    }

    override fun facesEnroll(
        file: ByteArray,
        contentType: String,
        filename: String,
        personName: String,
        personId: String?,
        eventId: UUID,
    ): FacesEnrollResult {
        val body = singleFileBody(file, contentType, filename).apply {
            add("person_name", personName)
            personId?.let { add("person_id", it) }
            add("event_id", eventId.toString())
        }
        return postAndUnwrap("/api/v1/faces/enroll", body, facesEnrollRef, retry = false)
    }

    override fun facesSearch(
        file: ByteArray,
        contentType: String,
        filename: String,
        eventId: UUID,
        threshold: Double,
        topK: Int,
    ): FacesSearchResult {
        val body = singleFileBody(file, contentType, filename)
        val path = "/api/v1/faces/search?event_id=$eventId&threshold=$threshold&top_k=$topK"
        return postAndUnwrap(path, body, facesSearchRef)
    }

    override fun bibsRecognize(
        file: ByteArray,
        contentType: String,
        filename: String,
        minChars: Int?,
    ): BibsRecognizeResult {
        val body = singleFileBody(file, contentType, filename)
        val path = if (minChars != null) "/api/v1/bibs/recognize?min_chars=$minChars" else "/api/v1/bibs/recognize"
        return postAndUnwrap(path, body, bibsRecognizeRef)
    }

    // Phase C bulk indexing — submit a whole event-batch as ONE async job.
    // Each file's filename is the photo id; ai-api echoes it back as `ref` so
    // results map without relying on positional order. Returns the job to poll.
    fun facesEnrollMega(files: List<NamedFile>, eventId: UUID): JobCreateResult {
        val body = multiFileBody(files).apply { add("event_id", eventId.toString()) }
        return postAndUnwrap("/api/v1/faces/enroll/mega", body, jobCreateRef, retry = false)
    }

    fun bibsRecognizeMega(files: List<NamedFile>): JobCreateResult {
        val body = multiFileBody(files)
        return postAndUnwrap("/api/v1/bibs/recognize/mega", body, jobCreateRef)
    }

    // Webhook subscription management (prod webhook-receiver path). Best-effort,
    // JSON not multipart, no retry wrapper — the startup runner that calls these
    // swallows failures (the poll backstop ingests regardless).
    fun listWebhookUrls(): List<String> {
        val raw = client.get().uri("/api/v1/webhooks").retrieve().body(String::class.java) ?: return emptyList()
        val webhooks = objectMapper.readTree(raw).path("data").path("webhooks")
        if (!webhooks.isArray) return emptyList()
        return webhooks.mapNotNull { wh -> wh.path("url").asText().takeIf { it.isNotBlank() } }
    }

    fun registerWebhook(url: String, events: List<String>, secret: String): Boolean {
        val raw = client.post()
            .uri("/api/v1/webhooks")
            .contentType(MediaType.APPLICATION_JSON)
            .body(mapOf("url" to url, "events" to events, "secret" to secret))
            .retrieve()
            .body(String::class.java) ?: return false
        return objectMapper.readTree(raw).path("success").asBoolean(false)
    }

    override fun deleteFacesPerson(personId: String) {
        withRetry("DELETE /faces/persons/$personId") {
            val raw = client.delete()
                .uri("/api/v1/faces/persons/{id}", personId)
                .retrieve()
                .body(String::class.java)
            val data = confirmedData(raw)
            if (data.path("deleted").booleanValue() != true || data.path("person_id").asText() != personId) {
                invalidConfirmation("Invalid face deletion confirmation")
            }
        }
    }

    override fun erasePhotoFaces(eventId: UUID, personId: String): FaceErasureResult {
        if (!props.enabled) invalidConfirmation("Face cleanup disabled")
        // The service validates photo attribution; this adapter independently
        // refuses a tenant-scoped deletion whose ID is outside the supplied event.
        if (listPersonsForEvent(eventId).none { it.id == personId }) return FaceErasureResult.ALREADY_ABSENT
        try {
            deleteFacesPerson(personId)
        } catch (ex: AiApiException) {
            if (ex.aiCode != ErrorCodes.NOT_FOUND) throw ex
        }
        if (listPersonsForEvent(eventId).any { it.id == personId }) invalidConfirmation("Face cleanup unconfirmed")
        return FaceErasureResult.DELETED
    }

    // GDPR bulk erasure: remove every ai-api person (and their face embeddings)
    // enrolled under one event in a single event-scoped call — used when a
    // backend deletes an event, instead of a per-photo deleteFacesPerson loop.
    override fun deleteFacesByEvent(eventId: UUID) {
        withRetry("DELETE /faces/persons?event_id=$eventId") {
            client.delete()
                .uri("/api/v1/faces/persons?event_id={eventId}", eventId.toString())
                .retrieve()
                .toBodilessEntity()
        }
    }

    // Every person ai-api holds for one event (tenant-scoped by the API key),
    // paged out fully. Invalid/incomplete envelopes never prove absence.
    override fun listPersonsForEvent(eventId: UUID): List<AiPersonRef> {
        if (!props.enabled) invalidConfirmation("Face cleanup disabled")
        val persons = mutableListOf<AiPersonRef>()
        var offset = 0
        var expectedTotal: Int? = null
        while (true) {
            val raw = client.get()
                .uri(
                    "/api/v1/faces/persons?event_id={e}&offset={o}&limit={l}",
                    eventId.toString(), offset, PERSON_PAGE_SIZE,
                )
                .retrieve()
                .body(String::class.java)
            val data = confirmedData(raw)
            val page = data.path("persons")
            val total = data.path("total")
            if (!page.isArray || !total.isIntegralNumber || total.intValue() < 0 ||
                data.path("offset").asInt(-1) != offset || data.path("limit").asInt(-1) != PERSON_PAGE_SIZE ||
                (expectedTotal != null && expectedTotal != total.intValue()) ||
                page.size() != minOf(PERSON_PAGE_SIZE, (total.intValue() - offset).coerceAtLeast(0))) {
                invalidConfirmation("Incomplete face listing")
            }
            expectedTotal = total.intValue()
            page.forEach { node ->
                val id = node.path("person_id").asText("")
                val createdAt = runCatching { OffsetDateTime.parse(node.path("created_at").asText("")) }.getOrNull()
                if (id.isBlank() || createdAt == null || node.path("event_id").asText() != eventId.toString() || persons.any { it.id == id }) {
                    invalidConfirmation("Invalid face listing")
                }
                persons.add(AiPersonRef(id, createdAt, node.path("person_name").takeIf { it.isTextual }?.asText()))
            }
            offset += page.size()
            if (offset == expectedTotal) break
        }
        return persons
    }

    private fun confirmedData(raw: String?): JsonNode {
        val root = runCatching { raw?.let(objectMapper::readTree) }.getOrNull()
            ?: invalidConfirmation("Invalid AI confirmation")
        if (!root.path("success").isBoolean || !root.path("success").booleanValue()) {
            if (root.path("error").path("code").asText() == ErrorCodes.NOT_FOUND) {
                throw AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, ErrorCodes.NOT_FOUND, "AI target not found")
            }
            invalidConfirmation("AI confirmation rejected")
        }
        return root.path("data").takeIf { it.isObject } ?: invalidConfirmation("Invalid AI confirmation")
    }

    private fun invalidConfirmation(reason: String): Nothing =
        throw AiApiException(HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.AI_API_UNAVAILABLE, reason)

    fun jobStatus(jobId: String, offset: Int? = null, limit: Int? = null): JobStatusResult {
        val query = buildList {
            if (offset != null) add("offset=$offset")
            if (limit != null) add("limit=$limit")
        }.joinToString("&")
        val path = if (query.isEmpty()) "/api/v1/jobs/$jobId" else "/api/v1/jobs/$jobId?$query"
        return withRetry("GET $path") {
            val raw = client.get().uri(path).retrieve().body(String::class.java)
            val data = confirmedData(raw)
            if (data.path("job_id").asText() != jobId) invalidConfirmation("Invalid job attribution")
            if (data.path("status").asText() == "completed") {
                val total = data.path("total_items")
                val result = data.path("result")
                if (!total.isIntegralNumber || total.intValue() < 0 ||
                    data.path("processed_items").asInt(-1) != total.intValue() ||
                    data.path("result_total").asInt(-1) != total.intValue() ||
                    data.path("result_offset").asInt(-1) != 0 || !result.isArray || result.size() != total.intValue()) {
                    invalidConfirmation("Incomplete terminal job results")
                }
            }
            objectMapper.treeToValue(data, JobStatusResult::class.java)
        }
    }

    fun jobStatusComplete(jobId: String): JobStatusResult = jobStatus(jobId, 0, 500)

    fun healthReady(): HealthReady = getAndUnwrap("/api/v1/health/ready", healthReadyRef)

    private fun <T> postAndUnwrap(path: String, body: MultiValueMap<String, Any>, ref: TypeReference<AiApiEnvelope<T>>, retry: Boolean = true): T =
        withRetry("POST $path", if (retry) props.maxRetries else 1) {
            val raw = client.post()
                .uri(path)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(String::class.java)
                ?: throw AiApiException(HttpStatus.BAD_GATEWAY, null, "ai-api returned empty body for $path")
            unwrap(raw, ref, path)
        }

    private fun <T> getAndUnwrap(path: String, ref: TypeReference<AiApiEnvelope<T>>): T =
        withRetry("GET $path") {
            val raw = client.get()
                .uri(path)
                .retrieve()
                .body(String::class.java)
                ?: throw AiApiException(HttpStatus.BAD_GATEWAY, null, "ai-api returned empty body for $path")
            unwrap(raw, ref, path)
        }

    private fun <T> unwrap(raw: String, ref: TypeReference<AiApiEnvelope<T>>, path: String): T {
        val envelope = try {
            objectMapper.readValue(raw, ref)
        } catch (ex: Exception) {
            throw AiApiException(HttpStatus.BAD_GATEWAY, null, "ai-api returned malformed JSON for $path", ex)
        }
        if (!envelope.success || envelope.data == null) {
            throw AiApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                aiCode = envelope.error?.code,
                message = envelope.error?.message ?: "ai-api call failed: $path",
            )
        }
        return envelope.data
    }

    private fun <T> withRetry(label: String, attempts: Int = props.maxRetries, block: () -> T): T {
        var attempt = 0
        var lastError: Throwable? = null
        while (attempt < attempts) {
            attempt++
            try {
                return block()
            } catch (ex: HttpServerErrorException) {
                lastError = ex
                log.warn("ai-api {} attempt {} failed with {}", label, attempt, ex.statusCode)
                if (attempt < attempts) sleepBackoff(attempt)
            } catch (ex: HttpClientErrorException) {
                if (ex.statusCode == HttpStatus.TOO_MANY_REQUESTS) {
                    lastError = ex
                    log.warn("ai-api {} attempt {} rate-limited", label, attempt)
                    if (attempt < attempts) sleepBackoff(attempt)
                } else {
                    throw AiApiException(
                        status = HttpStatus.valueOf(ex.statusCode.value()),
                        aiCode = null,
                        message = "ai-api $label failed: ${ex.message}",
                        cause = ex,
                    )
                }
            } catch (ex: ResourceAccessException) {
                lastError = ex
                log.warn("ai-api {} attempt {} I/O error: {}", label, attempt, ex.message)
                if (attempt < attempts) sleepBackoff(attempt)
            }
        }
        throw AiApiException(
            status = HttpStatus.SERVICE_UNAVAILABLE,
            aiCode = null,
            message = "ai-api $label failed after $attempts attempts",
            cause = lastError,
        )
    }

    private fun sleepBackoff(attempt: Int) {
        val backoff = props.backoffBaseMillis shl (attempt - 1).coerceAtMost(6)
        runCatching { Thread.sleep(backoff) }
    }

    private fun singleFileBody(file: ByteArray, contentType: String, filename: String): LinkedMultiValueMap<String, Any> {
        // The filename can be a client-supplied originalFilename and lands in
        // this part's Content-Disposition header — strip anything that could
        // smuggle header syntax (quotes, CR/LF, separators).
        val safeFilename = filename.replace(Regex("[^A-Za-z0-9._-]"), "_").ifEmpty { "file" }
        val resource = object : ByteArrayResource(file) {
            override fun getFilename(): String = safeFilename
        }
        val body = LinkedMultiValueMap<String, Any>()
        val partHeaders = org.springframework.http.HttpHeaders().apply {
            this.contentType = MediaType.parseMediaType(contentType)
        }
        body.add("file", org.springframework.http.HttpEntity(resource, partHeaders))
        return body
    }

    // Multi-file multipart for the mega endpoints: each file is a "files" part.
    // All parts reference their own byte[] (no copy); peak memory ≈ sum of bytes.
    private fun multiFileBody(files: List<NamedFile>): LinkedMultiValueMap<String, Any> {
        val body = LinkedMultiValueMap<String, Any>()
        files.forEach { nf ->
            val resource = object : ByteArrayResource(nf.bytes) {
                override fun getFilename(): String = nf.filename
            }
            val partHeaders = org.springframework.http.HttpHeaders().apply {
                contentType = MediaType.parseMediaType(nf.contentType)
            }
            body.add("files", org.springframework.http.HttpEntity(resource, partHeaders))
        }
        return body
    }

    private companion object {
        // ai-api caps /faces/persons limit at 200.
        const val PERSON_PAGE_SIZE = 200
        val facesDetectRef = object : TypeReference<AiApiEnvelope<FacesDetectResult>>() {}
        val facesEnrollRef = object : TypeReference<AiApiEnvelope<FacesEnrollResult>>() {}
        val facesSearchRef = object : TypeReference<AiApiEnvelope<FacesSearchResult>>() {}
        val bibsRecognizeRef = object : TypeReference<AiApiEnvelope<BibsRecognizeResult>>() {}
        val jobCreateRef = object : TypeReference<AiApiEnvelope<JobCreateResult>>() {}
        val healthReadyRef = object : TypeReference<AiApiEnvelope<HealthReady>>() {}
    }
}

// The worker echoes a UUID ref when supplied; optional null/absent refs use its
// integer index or the result position. Malformed supplied attribution is not a fallback.
internal fun jobResultPhoto(map: Map<*, *>, position: Int, photoIds: List<UUID>): UUID? {
    val ref = map["ref"]
    val id = if (ref != null) {
        if (ref !is String) return null
        runCatching { UUID.fromString(ref) }.getOrNull() ?: return null
    } else {
        val rawIndex = map["index"]
        val index = if (rawIndex == null) position else {
            val number = (rawIndex as? Number)?.toDouble() ?: return null
            if (!number.isFinite() || number % 1.0 != 0.0 || number < 0 || number >= photoIds.size) return null
            number.toInt()
        }
        photoIds.getOrNull(index) ?: return null
    }
    return id.takeIf { it in photoIds }
}

internal fun faceResultPhoto(item: Any?, position: Int, photoIds: List<UUID>): UUID? {
    val map = item as? Map<*, *> ?: return null
    val person = map["person_id"]
    val validPerson = map.containsKey("person_id") && (person == null || person is String && person.isNotBlank())
    val validError = (map["error"] as? String)?.isNotBlank() == true
    if (!validPerson && !validError) return null
    return jobResultPhoto(map, position, photoIds)
}
