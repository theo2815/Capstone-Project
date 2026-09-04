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
data class AiPersonRef(val id: String, val createdAt: OffsetDateTime)

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
        return postAndUnwrap("/api/v1/faces/enroll", body, facesEnrollRef)
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
        return postAndUnwrap("/api/v1/faces/enroll/mega", body, jobCreateRef)
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
            client.delete()
                .uri("/api/v1/faces/persons/{id}", personId)
                .retrieve()
                .toBodilessEntity()
        }
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
    // paged out fully. Best-effort, tree-parsed like the webhook helpers — the
    // reaper that calls this catches failures and retries on its next sweep.
    override fun listPersonsForEvent(eventId: UUID): List<AiPersonRef> {
        val persons = mutableListOf<AiPersonRef>()
        var offset = 0
        while (true) {
            val raw = client.get()
                .uri(
                    "/api/v1/faces/persons?event_id={e}&offset={o}&limit={l}",
                    eventId.toString(), offset, PERSON_PAGE_SIZE,
                )
                .retrieve()
                .body(String::class.java) ?: break
            val data = objectMapper.readTree(raw).path("data")
            val page = data.path("persons")
            if (!page.isArray || page.isEmpty) break
            page.forEach { node ->
                val id = node.path("person_id").asText("")
                val createdAt = runCatching { OffsetDateTime.parse(node.path("created_at").asText("")) }.getOrNull()
                if (id.isNotBlank() && createdAt != null) persons.add(AiPersonRef(id, createdAt))
            }
            offset += PERSON_PAGE_SIZE
            if (offset >= data.path("total").asInt(persons.size)) break
        }
        return persons
    }

    fun jobStatus(jobId: String, offset: Int? = null, limit: Int? = null): JobStatusResult {
        val query = buildList {
            if (offset != null) add("offset=$offset")
            if (limit != null) add("limit=$limit")
        }.joinToString("&")
        val path = if (query.isEmpty()) "/api/v1/jobs/$jobId" else "/api/v1/jobs/$jobId?$query"
        return getAndUnwrap(path, jobStatusRef)
    }

    fun healthReady(): HealthReady = getAndUnwrap("/api/v1/health/ready", healthReadyRef)

    private fun <T> postAndUnwrap(path: String, body: MultiValueMap<String, Any>, ref: TypeReference<AiApiEnvelope<T>>): T =
        withRetry("POST $path") {
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

    private fun <T> withRetry(label: String, block: () -> T): T {
        var attempt = 0
        var lastError: Throwable? = null
        while (attempt < props.maxRetries) {
            attempt++
            try {
                return block()
            } catch (ex: HttpServerErrorException) {
                lastError = ex
                log.warn("ai-api {} attempt {} failed with {}", label, attempt, ex.statusCode)
                sleepBackoff(attempt)
            } catch (ex: HttpClientErrorException) {
                if (ex.statusCode == HttpStatus.TOO_MANY_REQUESTS) {
                    lastError = ex
                    log.warn("ai-api {} attempt {} rate-limited", label, attempt)
                    sleepBackoff(attempt)
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
                sleepBackoff(attempt)
            }
        }
        throw AiApiException(
            status = HttpStatus.SERVICE_UNAVAILABLE,
            aiCode = null,
            message = "ai-api $label failed after ${props.maxRetries} retries",
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
        val jobStatusRef = object : TypeReference<AiApiEnvelope<JobStatusResult>>() {}
        val healthReadyRef = object : TypeReference<AiApiEnvelope<HealthReady>>() {}
    }
}
