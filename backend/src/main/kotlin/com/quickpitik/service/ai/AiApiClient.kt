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
import java.util.UUID

@Service
class AiApiClient(
    @Qualifier("aiApiRestClient") private val client: RestClient,
    private val props: AiApiProperties,
    private val objectMapper: ObjectMapper,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun facesDetect(file: ByteArray, contentType: String, filename: String): FacesDetectResult {
        val body = singleFileBody(file, contentType, filename)
        return postAndUnwrap("/api/v1/faces/detect", body, facesDetectRef)
    }

    fun facesEnroll(
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

    fun facesSearch(
        file: ByteArray,
        contentType: String,
        filename: String,
        eventId: UUID,
        threshold: Double = props.faceMatchThresholdDefault,
        topK: Int = props.faceTopKDefault,
    ): FacesSearchResult {
        val body = singleFileBody(file, contentType, filename)
        val path = "/api/v1/faces/search?event_id=$eventId&threshold=$threshold&top_k=$topK"
        return postAndUnwrap(path, body, facesSearchRef)
    }

    fun bibsRecognize(
        file: ByteArray,
        contentType: String,
        filename: String,
        minChars: Int? = null,
    ): BibsRecognizeResult {
        val body = singleFileBody(file, contentType, filename)
        val path = if (minChars != null) "/api/v1/bibs/recognize?min_chars=$minChars" else "/api/v1/bibs/recognize"
        return postAndUnwrap(path, body, bibsRecognizeRef)
    }

    fun deleteFacesPerson(personId: String) {
        withRetry("DELETE /faces/persons/$personId") {
            client.delete()
                .uri("/api/v1/faces/persons/{id}", personId)
                .retrieve()
                .toBodilessEntity()
        }
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
        val resource = object : ByteArrayResource(file) {
            override fun getFilename(): String = filename
        }
        val body = LinkedMultiValueMap<String, Any>()
        val partHeaders = org.springframework.http.HttpHeaders().apply {
            this.contentType = MediaType.parseMediaType(contentType)
        }
        body.add("file", org.springframework.http.HttpEntity(resource, partHeaders))
        return body
    }

    private companion object {
        val facesDetectRef = object : TypeReference<AiApiEnvelope<FacesDetectResult>>() {}
        val facesEnrollRef = object : TypeReference<AiApiEnvelope<FacesEnrollResult>>() {}
        val facesSearchRef = object : TypeReference<AiApiEnvelope<FacesSearchResult>>() {}
        val bibsRecognizeRef = object : TypeReference<AiApiEnvelope<BibsRecognizeResult>>() {}
        val jobStatusRef = object : TypeReference<AiApiEnvelope<JobStatusResult>>() {}
        val healthReadyRef = object : TypeReference<AiApiEnvelope<HealthReady>>() {}
    }
}
