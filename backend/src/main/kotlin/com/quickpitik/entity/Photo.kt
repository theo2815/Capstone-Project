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

    @Column(name = "blur_score", precision = 10, scale = 4)
    var blurScore: BigDecimal? = null,

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

    @Column(name = "price_php", nullable = false, precision = 12, scale = 2)
    var pricePhp: BigDecimal,

    @Column(name = "alt_text", columnDefinition = "TEXT")
    var altText: String? = null,

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
