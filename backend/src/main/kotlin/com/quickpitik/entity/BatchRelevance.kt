package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

// PE-1: one batch's scored relevance snapshot (V100 batch_relevance). A batch is one
// event_photographer row. Rows are made by BatchRelevanceRepository.ensureBatch and written only
// by service/relevance. NULL counts and flags mean "not scored" or "not decided".
@Entity
@Table(name = "batch_relevance")
class BatchRelevance(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "event_id", nullable = false, updatable = false)
    val eventId: UUID,

    @Column(name = "photographer_id", nullable = false, updatable = false)
    val photographerId: UUID,

    // Set by the database default.
    @Column(name = "created_at", insertable = false, updatable = false)
    val createdAt: OffsetDateTime? = null,

    @Column(name = "last_swept_at") var lastSweptAt: OffsetDateTime? = null,
    @Column(name = "scored_at") var scoredAt: OffsetDateTime? = null,
    @Column(name = "scored_upload_at") var scoredUploadAt: OffsetDateTime? = null,
    @Column(name = "scored_event_date") var scoredEventDate: LocalDate? = null,
    @Column(name = "final_scored_at") var finalScoredAt: OffsetDateTime? = null,
    @Column(name = "photos_live") var photosLive: Int? = null,
    @Column(name = "photos_processing") var photosProcessing: Int? = null,
    @Column(name = "lookups") var lookups: Int? = null,
    @Column(name = "tight_hits") var tightHits: Int? = null,
    @Column(name = "weak_hits") var weakHits: Int? = null,
    @Column(name = "exact_matches") var exactMatches: Int? = null,
    @Column(name = "xmp_matches") var xmpMatches: Int? = null,
    @Column(name = "dated_photos") var datedPhotos: Int? = null,
    @Column(name = "dated_outside") var datedOutside: Int? = null,
    @Column(name = "metadata_pending") var metadataPending: Int? = null,
    @Column(name = "indexed_photos") var indexedPhotos: Int? = null,
    @Column(name = "bibbed_photos") var bibbedPhotos: Int? = null,
    @Column(name = "bib_overlap_photos") var bibOverlapPhotos: Int? = null,
    @Column(name = "copy_low") var copyLow: Boolean? = null,
    @Column(name = "date_low") var dateLow: Boolean? = null,
    // NULL = not Low. Version 1 has only LOW (D22).
    @Column(name = "level", length = 8) var level: String? = null,
    // PE-2.1: the relevance flag this batch raised or extended, and when.
    @Column(name = "flag_id") var flagId: UUID? = null,
    @Column(name = "flagged_at") var flaggedAt: OffsetDateTime? = null,
)
