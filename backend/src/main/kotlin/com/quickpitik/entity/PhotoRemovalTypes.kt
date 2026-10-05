package com.quickpitik.entity

enum class RemovalRequestStatus { DRAFT, OPEN, RESOLVING, RESOLVED, CANCELLED }
enum class RemovalItemState { CANDIDATE, VERIFIED, EXCLUDED, REMOVAL_AUTHORIZED, REMOVED, RETAINED, RETAINED_RESTRICTED }
enum class PhotographerRemovalResponse { AGREE, OBJECT, ALREADY_DELETED }
enum class RemovalDecision { REMOVE, RETAIN, MORE_EVIDENCE, RETAIN_RESTRICTED }
enum class RemovalState { NONE, REVIEW, REMOVED }
enum class RemovalMatchSource { SELFIE, BIB, MANUAL }
enum class RemovalActorType { ADMIN, PHOTOGRAPHER, SYSTEM, SUPPORT }
enum class RemovalEvidenceCleanupState { NONE, PRESENT, PENDING, DELETED, RETRY_REQUIRED }
