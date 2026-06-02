package com.quickpitik.entity

// Lifecycle of a photo's ai-api indexing (face enroll + bib OCR), which runs
// asynchronously off the upload request. See PhotoIndexingService.
//
// PENDING  — uploaded, not yet indexed (or queued for retry).
// BATCHING — submitted to ai-api inside a mega job (Phase C batch mode); awaiting
//            ingest. Excluded from the per-photo sweep; settled by batch ingest.
// INDEXED  — face + bib both completed (or found nothing — that still counts).
// PARTIAL  — one of face/bib succeeded, the other hit a transient ai-api error.
// FAILED   — indexing failed and exhausted its retry attempts.
// SKIPPED  — ai-api disabled (AI_API_ENABLED=false); never attempted.
enum class IndexingStatus {
    PENDING,
    BATCHING,
    INDEXED,
    PARTIAL,
    FAILED,
    SKIPPED,
}
