package com.quickpitik.entity

enum class RemovalRequestStatus { DRAFT, OPEN, RESOLVING, RESOLVED, CANCELLED }
enum class RemovalItemState { CANDIDATE, VERIFIED, EXCLUDED, REMOVAL_AUTHORIZED, REMOVED, RETAINED, RETAINED_RESTRICTED }
enum class PhotographerRemovalResponse { AGREE, OBJECT, ALREADY_DELETED }
enum class RemovalDecision { REMOVE, RETAIN, MORE_EVIDENCE, RETAIN_RESTRICTED }
enum class RemovalState { NONE, REVIEW, REMOVED }
enum class RemovalMatchSource { SELFIE, BIB, MANUAL }
enum class RemovalActorType { ADMIN, PHOTOGRAPHER, SYSTEM, SUPPORT }
enum class RemovalEvidenceCleanupState { NONE, PRESENT, PENDING, DELETED, RETRY_REQUIRED }

// A runner's live-check result for a deletion request; a fail is never stored.
enum class RunnerLivenessBand { PASS, UNCERTAIN }

// Why a runner request's matches were held back or need a closer look; stored by name in risk_flags.
enum class RunnerRiskFlag {
    LIVENESS_UNCERTAIN, UNVERIFIED_EMAIL, PRIOR_REJECTION, POSSIBLY_TRUNCATED, OVER_REQUEST_CAP,
    PHOTOGRAPHER_BUDGET, AUTO_HIDE_OFF, GROUP_PHOTOS, MATCHING_UNAVAILABLE,
}
