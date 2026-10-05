package com.quickpitik.service.removal

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Component

@Component
class PhotoRemovalTransitions {
    fun ensureVersion(request: PhotoRemovalRequest, expectedVersion: Long) {
        if (expectedVersion < 0 || request.version != expectedVersion) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.CONFLICT, "Photo removal request changed; refresh before retrying")
        }
    }

    // Caller owns persistence, actor/time stamping and the same-transaction history event.
    fun apply(item: PhotoRemovalItem, decision: RemovalDecision, reason: String) {
        if (item.state !in setOf(RemovalItemState.VERIFIED, RemovalItemState.RETAINED_RESTRICTED)) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION, "Cannot decide this photo removal item")
        }
        if (reason.isBlank() || (decision == RemovalDecision.RETAIN_RESTRICTED && item.retentionReviewAt == null)) {
            throw ApiException(HttpStatus.BAD_REQUEST, ErrorCodes.VALIDATION_ERROR, "A reason and any retention review date are required")
        }
        item.state = when (decision) {
            RemovalDecision.REMOVE -> RemovalItemState.REMOVAL_AUTHORIZED
            RemovalDecision.RETAIN -> RemovalItemState.RETAINED
            RemovalDecision.MORE_EVIDENCE -> RemovalItemState.VERIFIED
            RemovalDecision.RETAIN_RESTRICTED -> RemovalItemState.RETAINED_RESTRICTED
        }
        item.adminDecision = decision
        item.adminDecisionReason = reason
    }
}
