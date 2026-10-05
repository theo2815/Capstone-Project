package com.quickpitik.service.removal

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.GlobalExceptionHandler
import jakarta.validation.Validation
import org.junit.jupiter.api.Test
import java.time.Instant
import java.util.UUID
import kotlin.test.*

class PhotoRemovalTransitionsTest {
    private val transitions = PhotoRemovalTransitions()
    private fun item(state: RemovalItemState = RemovalItemState.VERIFIED) = PhotoRemovalItem(
        requestId = UUID.randomUUID(), photoIdSnapshot = UUID.randomUUID(), state = state,
        matchSource = RemovalMatchSource.MANUAL,
    )

    private data class Failure(val status: Int)
    private fun applyRetainToAuthorizedItem(): Failure {
        val selected = item()
        transitions.apply(selected, RemovalDecision.REMOVE, "Verified support request")
        val error = assertFailsWith<ApiException> {
            transitions.apply(selected, RemovalDecision.RETAIN, "Changed my mind")
        }
        assertEquals(RemovalItemState.REMOVAL_AUTHORIZED, selected.state)
        assertEquals(RemovalDecision.REMOVE, selected.adminDecision)
        assertEquals("Verified support request", selected.adminDecisionReason)
        return Failure(GlobalExceptionHandler().handleApi(error).statusCode.value())
    }

    @Test fun `authorizationCannotBeReversed`() {
        assertEquals(409, applyRetainToAuthorizedItem().status)
        for (state in listOf(RemovalItemState.REMOVAL_AUTHORIZED, RemovalItemState.REMOVED,
            RemovalItemState.EXCLUDED, RemovalItemState.RETAINED)) {
            for (decision in RemovalDecision.entries) {
                val selected = item(state).also { it.retentionReviewAt = Instant.parse("2026-10-08T00:00:00Z") }
                assertEquals(409, assertFailsWith<ApiException> {
                    transitions.apply(selected, decision, "Another decision")
                }.status.value())
                assertEquals(state, selected.state)
                assertNull(selected.adminDecision)
            }
        }
    }

    @Test fun `restricted retention remains reviewable with each admin decision`() {
        for ((decision, expected) in listOf(
            RemovalDecision.REMOVE to RemovalItemState.REMOVAL_AUTHORIZED,
            RemovalDecision.RETAIN to RemovalItemState.RETAINED,
            RemovalDecision.MORE_EVIDENCE to RemovalItemState.VERIFIED,
            RemovalDecision.RETAIN_RESTRICTED to RemovalItemState.RETAINED_RESTRICTED,
        )) {
            val selected = item(RemovalItemState.RETAINED_RESTRICTED).also {
                it.retentionReviewAt = Instant.parse("2026-10-08T00:00:00Z")
            }
            transitions.apply(selected, decision, "Follow-up decision")
            assertEquals(expected, selected.state)
            assertEquals(decision, selected.adminDecision)
            assertEquals("Follow-up decision", selected.adminDecisionReason)
        }
    }

    @Test fun `only verified items accept admin decisions`() {
        for (decision in RemovalDecision.entries) {
            val selected = item(RemovalItemState.CANDIDATE)
            assertEquals(409, assertFailsWith<ApiException> {
                transitions.apply(selected, decision, "Not verified")
            }.status.value())
            assertEquals(RemovalItemState.CANDIDATE, selected.state)
        }
    }

    @Test fun `decisions retain their distinct outcomes and reasons`() {
        for ((decision, expected) in listOf(
            RemovalDecision.REMOVE to RemovalItemState.REMOVAL_AUTHORIZED,
            RemovalDecision.RETAIN to RemovalItemState.RETAINED,
            RemovalDecision.MORE_EVIDENCE to RemovalItemState.VERIFIED,
            RemovalDecision.RETAIN_RESTRICTED to RemovalItemState.RETAINED_RESTRICTED,
        )) {
            val selected = item().also { it.retentionReviewAt = Instant.parse("2026-10-08T00:00:00Z") }
            transitions.apply(selected, decision, "Documented reason")
            assertEquals(expected, selected.state)
            assertEquals(decision, selected.adminDecision)
            assertEquals("Documented reason", selected.adminDecisionReason)
        }
    }

    @Test fun `blank reasons and undated restricted retention cannot mutate an item`() {
        for (reason in listOf("", " ", "\t\n")) {
            val selected = item()
            assertEquals(400, assertFailsWith<ApiException> {
                transitions.apply(selected, RemovalDecision.REMOVE, reason)
            }.status.value())
            assertEquals(RemovalItemState.VERIFIED, selected.state)
            assertNull(selected.adminDecision)
        }
        val selected = item()
        assertEquals(400, assertFailsWith<ApiException> {
            transitions.apply(selected, RemovalDecision.RETAIN_RESTRICTED, "Exception")
        }.status.value())
        assertEquals(RemovalItemState.VERIFIED, selected.state)
    }

    @Test fun `stale and negative versions conflict before mutation`() {
        val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Run",
            requesterContact = "Support-held contact", supportReference = "Support 1", requesterReason = "Removal",
            publicToOwnerReason = "Privacy request", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(),
            version = 7)
        transitions.ensureVersion(request, 7)
        for (version in listOf(-1L, 0L, 6L, 8L)) {
            assertEquals(409, assertFailsWith<ApiException> { transitions.ensureVersion(request, version) }.status.value())
        }
        assertEquals(7L, request.version)
    }

    @Test fun `photographer JSON has no intake evidence or verification fields`() {
        val dto = PhotographerRemovalRequestDto(
            id = UUID.randomUUID(), event = RemovalEventDto(UUID.randomUUID(), "Cebu run"),
            status = RemovalRequestStatus.OPEN, version = 1, createdAt = Instant.parse("2026-10-05T00:00:00Z"),
            updatedAt = Instant.parse("2026-10-05T00:00:00Z"), sanitizedReason = "Verified privacy request",
            items = emptyList(), reviews = emptyList(),
        )
        val json = jacksonObjectMapper().findAndRegisterModules().valueToTree<com.fasterxml.jackson.databind.JsonNode>(dto)
        assertEquals("Verified privacy request", json["sanitizedReason"].asText())
        for (field in listOf("requesterContact", "requesterUserId", "supportReference", "requesterReason",
            "evidenceKey", "evidenceContentType", "outstandingEvidenceKeys", "verificationMethod", "verificationNote")) {
            assertFalse(json.has(field), "Photographer projection disclosed $field")
        }
    }

    @Test fun `command DTOs reject blank reasons and more than one hundred items`() {
        Validation.buildDefaultValidatorFactory().use { factory ->
            val validator = factory.validator
            val create = CreateRemovalRequestCommand(UUID.randomUUID(), " ", " ", reason = " ", commandId = UUID.randomUUID())
            assertEquals(setOf("supportReference", "requesterContact", "reason"), validator.validate(create).map { it.propertyPath.toString() }.toSet())
            val respond = RespondToRemovalCommand(UUID.randomUUID(), 0, List(101) { UUID.randomUUID() }, PhotographerRemovalResponse.AGREE)
            assertTrue(validator.validate(respond).any { it.propertyPath.toString() == "itemIds" })
            val decision = DecideRemovalCommand(UUID.randomUUID(), 0, listOf(RemovalItemDecision(UUID.randomUUID(), RemovalDecision.REMOVE, " ")))
            assertTrue(validator.validate(decision).any { it.propertyPath.toString() == "decisions[0].reason" })
        }
    }
}
