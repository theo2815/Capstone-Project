package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// RV-1.1: the values the CHECK constraints of relevance_cases allow (V105). The columns stay strings.
enum class CaseStage { SAMPLING, DECIDING, DECIDED, APPLIED, CLOSED }
enum class CaseVerdict { FITS, MIXED, JUNK_DUMP, WRONG_EVENT, COPIED, CANT_TELL }
enum class CaseBand { UNDER, MIXED, OVER, SMALL, UNKNOWN }
enum class CaseAction { NONE, HIDE_JUNK, HIDE_ALL }
enum class ScanState { NONE, QUEUED, RUNNING, FINISHED }

// RV-1.1 (Photo Relevance v2): one check of one batch (V105 relevance_cases). Written only by
// service/relevance. A batch has at most one case in SAMPLING, DECIDING or DECIDED, and at most one
// APPLIED (its current case); both are unique indexes. NULL counts mean "not measured yet".
@Entity
@Table(name = "relevance_cases")
class RelevanceCase(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "batch_id", nullable = false, updatable = false)
    val batchId: UUID,

    // Set by the database default.
    @Column(name = "created_at", insertable = false, updatable = false)
    val createdAt: OffsetDateTime? = null,

    @Column(name = "trigger", nullable = false, updatable = false, length = 8)
    val trigger: String,

    @Column(name = "upload_at", nullable = false, updatable = false)
    val uploadAt: OffsetDateTime,

    @Column(name = "batch_photos", nullable = false, updatable = false)
    val batchPhotos: Int,

    @Column(name = "stage", nullable = false, length = 10) var stage: String = CaseStage.SAMPLING.name,
    @Column(name = "sample_round", nullable = false) var sampleRound: Short = 1,
    @Column(name = "sample_size") var sampleSize: Int? = null,
    @Column(name = "measured") var measured: Int? = null,
    @Column(name = "blurry") var blurry: Int? = null,
    @Column(name = "dark") var dark: Int? = null,
    @Column(name = "not_race") var notRace: Int? = null,
    @Column(name = "junk") var junk: Int? = null,
    @Column(name = "words_photos") var wordsPhotos: Int? = null,
    @Column(name = "band", length = 8) var band: String? = null,
    @Column(name = "copy_low") var copyLow: Boolean? = null,
    @Column(name = "date_low") var dateLow: Boolean? = null,
    @Column(name = "decide_attempts", nullable = false) var decideAttempts: Short = 0,
    // One of the six verdict values or NULL, never the model's raw text.
    @Column(name = "model_verdict", length = 12) var modelVerdict: String? = null,
    @Column(name = "verdict", length = 12) var verdict: String? = null,
    @Column(name = "verdict_note", length = 40) var verdictNote: String? = null,
    @Column(name = "admin_reason", columnDefinition = "TEXT") var adminReason: String? = null,
    @Column(name = "evidence_used", columnDefinition = "TEXT") var evidenceUsed: String? = null,
    @Column(name = "summary_sent", columnDefinition = "TEXT") var summarySent: String? = null,
    @Column(name = "model", length = 60) var model: String? = null,
    @Column(name = "prompt_version", length = 12) var promptVersion: String? = null,
    @Column(name = "decided_at") var decidedAt: OffsetDateTime? = null,
    @Column(name = "applied_mode", length = 12) var appliedMode: String? = null,
    @Column(name = "action", length = 10) var action: String? = null,
    @Column(name = "applied_at") var appliedAt: OffsetDateTime? = null,
    @Column(name = "closed_at") var closedAt: OffsetDateTime? = null,
    @Column(name = "close_reason", length = 24) var closeReason: String? = null,
    // RV-2.1 (V106): what acting needs. Nothing writes these yet.
    @Column(name = "hidden_photos", nullable = false) var hiddenPhotos: Int = 0,
    @Column(name = "hide_finished_at") var hideFinishedAt: OffsetDateTime? = null,
    @Column(name = "scan_state", nullable = false, length = 8) var scanState: String = ScanState.NONE.name,
    @Column(name = "scan_served_at") var scanServedAt: OffsetDateTime? = null,
    @Column(name = "scan_finished_at") var scanFinishedAt: OffsetDateTime? = null,
    @Column(name = "scan_measured") var scanMeasured: Int? = null,
    @Column(name = "scan_junk") var scanJunk: Int? = null,
    @Column(name = "scan_blurry") var scanBlurry: Int? = null,
    @Column(name = "flipped_at") var flippedAt: OffsetDateTime? = null,
    @Column(name = "notice_at") var noticeAt: OffsetDateTime? = null,
    @Column(name = "confirmed_at") var confirmedAt: OffsetDateTime? = null,
    @Column(name = "closed_by") var closedBy: UUID? = null,
    @Column(name = "hold_flag_id") var holdFlagId: UUID? = null,
)
