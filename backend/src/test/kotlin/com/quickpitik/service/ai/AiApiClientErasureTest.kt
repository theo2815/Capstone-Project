package com.quickpitik.service.ai

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.config.AiApiProperties
import org.junit.jupiter.api.Test
import org.springframework.http.HttpMethod
import org.springframework.http.MediaType
import org.springframework.test.web.client.MockRestServiceServer
import org.springframework.test.web.client.match.MockRestRequestMatchers.method
import org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo
import org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess
import org.springframework.web.client.RestClient
import java.util.UUID
import kotlin.test.assertFalse
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class AiApiClientErasureTest {
    private val event = UUID.randomUUID()
    private val person = UUID.randomUUID().toString()
    private val builder = RestClient.builder().baseUrl("http://ai.test")
    private val server = MockRestServiceServer.bindTo(builder).build()
    private val client = AiApiClient(builder.build(), AiApiProperties(enabled = true, maxRetries = 1), jacksonObjectMapper())

    @Test fun `http200ErrorEnvelopeCannotCompleteErasure`() {
        server.expect(requestTo("http://ai.test/api/v1/faces/persons/$person"))
            .andExpect(method(HttpMethod.DELETE))
            .andRespond(withSuccess("""{"success":false,"error":{"code":"MODEL_UNAVAILABLE","message":"private provider text"}}""", MediaType.APPLICATION_JSON))

        val cleanupRemainsRetryRequired = runCatching { client.deleteFacesPerson(person) }.isFailure

        assertFalse(!cleanupRemainsRetryRequired)
        server.verify()
    }

    @Test fun `missing deletion data cannot prove absence`() {
        server.expect(requestTo("http://ai.test/api/v1/faces/persons/$person"))
            .andRespond(withSuccess("""{"success":true}""", MediaType.APPLICATION_JSON))
        assertFalse(runCatching { client.deleteFacesPerson(person) }.isSuccess)
        server.verify()
    }

    @Test fun `error listing cannot prove an empty event`() {
        listing("""{"success":false,"error":{"code":"FORBIDDEN","message":"private"}}""")
        assertFalse(runCatching { client.listPersonsForEvent(event) }.isSuccess)
        server.verify()
    }

    @Test fun `truncated listing cannot prove absence`() {
        listing("""{"success":true,"data":{"persons":[],"total":1,"offset":0,"limit":200}}""")
        assertFalse(runCatching { client.listPersonsForEvent(event) }.isSuccess)
        server.verify()
    }

    @Test fun `malformed person in listing cannot disappear silently`() {
        listing("""{"success":true,"data":{"persons":[{"person_id":"$person","person_name":"photo","created_at":"invalid"}],"total":1,"offset":0,"limit":200}}""")
        assertFalse(runCatching { client.listPersonsForEvent(event) }.isSuccess)
        server.verify()
    }

    private fun listing(body: String) {
        server.expect(requestTo("http://ai.test/api/v1/faces/persons?event_id=$event&offset=0&limit=200"))
            .andExpect(method(HttpMethod.GET)).andRespond(withSuccess(body, MediaType.APPLICATION_JSON))
    }

    @Test fun `truncated completed job cannot prove writer completion`() {
        server.expect(requestTo("http://ai.test/api/v1/jobs/face-job?offset=0&limit=500"))
            .andRespond(withSuccess("""{"success":true,"data":{"job_id":"face-job","status":"completed","progress":1.0,"total_items":2,"processed_items":2,"result_total":2,"result_offset":0,"result_limit":500,"result":[{"ref":"photo-1","person_id":"$person"}]}}""", MediaType.APPLICATION_JSON))
        assertFailsWith<AiApiException> { client.jobStatusComplete("face-job") }
        server.verify()
    }

    @Test fun `complete job retains its attributable results`() {
        server.expect(requestTo("http://ai.test/api/v1/jobs/face-job?offset=0&limit=500"))
            .andRespond(withSuccess("""{"success":true,"data":{"job_id":"face-job","status":"completed","progress":1.0,"total_items":1,"processed_items":1,"result_total":1,"result_offset":0,"result_limit":500,"result":[{"ref":"photo-1","person_id":"$person"}]}}""", MediaType.APPLICATION_JSON))
        assertEquals(person, (client.jobStatusComplete("face-job").result!!.single() as Map<*, *>)["person_id"])
        server.verify()
    }
}
