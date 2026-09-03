package com.quickpitik.entity

import jakarta.persistence.CollectionTable
import jakarta.persistence.Column
import jakarta.persistence.ElementCollection
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.FetchType
import jakarta.persistence.Id
import jakarta.persistence.JoinColumn
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

@Entity
@Table(name = "photos")
class Photo(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "event_id", nullable = false)
    var eventId: UUID,

    @Column(name = "photographer_id")
    var photographerId: UUID? = null,

    @Column(name = "s3_key", nullable = false, length = 512)
    var s3Key: String,

    @Column(name = "thumbnail_s3_key", length = 512)
    var thumbnailS3Key: String? = null,

    @Column(name = "watermark_s3_key", length = 512)
    var watermarkS3Key: String? = null,

    // SHA-256 hexdigest of the ORIGINAL uploaded bytes (pre-watermark) — the
    // identity key for per-photographer duplicate detection. Partial unique
    // index (photographer_id, content_hash) lives in migration V24. Nullable:
    // rows uploaded before V24 have no hash and are excluded from the index.
    @Column(name = "content_hash", length = 64)
    var contentHash: String? = null,

    @Column(name = "span", nullable = false, length = 20)
    @JdbcTypeCode(SqlTypes.VARCHAR)
    var spanWire: String = PhotoSpan.DEFAULT.wire,

    @Column(name = "tone", nullable = false)
    var tone: Int = 0,

    @Column(name = "km", precision = 5, scale = 2)
    var km: BigDecimal? = null,

    @Column(name = "captured_at")
    var capturedAt: OffsetDateTime? = null,

    @Column(name = "uploaded_at", nullable = false)
    var uploadedAt: OffsetDateTime = OffsetDateTime.now(),

    @Enumerated(EnumType.STRING)
    @Column(name = "status", nullable = false, length = 20)
    var status: PhotoStatus = PhotoStatus.LIVE,

    // When the photo became visible to runners. Upload time is not enough:
    // watermarking is asynchronous, so a PROCESSING row can predate the
    // gallery snapshot while becoming LIVE after it.
    @Column(name = "published_at")
    var publishedAt: OffsetDateTime? = if (status == PhotoStatus.LIVE) uploadedAt else null,

    @Column(name = "price_php", nullable = false, precision = 12, scale = 2)
    var pricePhp: BigDecimal,

    @Column(name = "alt_text", columnDefinition = "TEXT")
    var altText: String? = null,

    // Async indexing state (face enroll + bib OCR via ai-api). Written by
    // PhotoIndexingService off the upload request; a @Scheduled sweep re-drives
    // PENDING/FAILED rows. See migration V21.
    @Enumerated(EnumType.STRING)
    @Column(name = "indexing_status", nullable = false, length = 16)
    var indexingStatus: IndexingStatus = IndexingStatus.PENDING,

    @Column(name = "indexed_at")
    var indexedAt: OffsetDateTime? = null,

    @Column(name = "indexing_attempts", nullable = false)
    var indexingAttempts: Int = 0,

    // Async-watermark retry budget (V36). Photos are created PROCESSING and
    // flipped LIVE by PhotoWatermarkTrigger; only semantic failures
    // (undecodable bytes) consume this — transport failures leave it intact so
    // the reconcile sweep keeps re-driving them.
    @Column(name = "processing_attempts", nullable = false)
    var processingAttempts: Int = 0,

    // 64-bit perceptual hash of the MARKED preview (watermark.jpg) — the leak
    // surface a screenshot is matched against by POST /public/photos/verify
    // (V42). Null until PhotoWatermarkService computes it at the LIVE flip or
    // the reconcile sweep backfills it.
    @Column(name = "phash")
    var phash: Long? = null,

    // Fingerprints of the frame BEFORE the mark was drawn — full frame and the
    // middle 60% crop (V43) — so a cleaned or runner-cropped copy still
    // attributes. Null on pre-V43 rows; never backfilled (the clean render is
    // gone once the preview exists).
    @Column(name = "phash_clean")
    var phashClean: Long? = null,

    @Column(name = "phash_centre")
    var phashCentre: Long? = null,

    @Column(name = "indexing_error", columnDefinition = "TEXT")
    var indexingError: String? = null,

    // Which FaceBibProvider produced the stored face/bib results (V33):
    // "ai_api" or "rekognition". Their person-id spaces are incompatible, so a
    // provider flip leaves stale rows detectable — and re-drivable via the
    // admin reindex endpoint — by this stamp.
    @Column(name = "indexed_provider", length = 16)
    var indexedProvider: String? = null,

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(
        name = "photo_bibs",
        joinColumns = [JoinColumn(name = "photo_id")],
    )
    var bibs: MutableSet<PhotoBibEmbed> = mutableSetOf(),

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(
        name = "photo_face_persons",
        joinColumns = [JoinColumn(name = "photo_id")],
    )
    var facePersons: MutableSet<PhotoFacePersonEmbed> = mutableSetOf(),
) {
    var span: PhotoSpan
        get() = PhotoSpan.fromWire(spanWire)
        set(value) {
            spanWire = value.wire
        }
}
