package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable
import jakarta.persistence.EmbeddedId
import jakarta.persistence.Entity
import jakarta.persistence.Table
import java.io.Serializable
import java.time.OffsetDateTime
import java.util.UUID

@Embeddable
data class DownloadGrantId(
    @Column(name = "order_id", nullable = false)
    var orderId: UUID,

    @Column(name = "photo_id", nullable = false)
    var photoId: UUID,
) : Serializable

@Entity
@Table(name = "download_grants")
class DownloadGrant(
    @EmbeddedId
    val id: DownloadGrantId,

    @Column(name = "granted_until", nullable = false)
    var grantedUntil: OffsetDateTime,

    @Column(name = "last_downloaded_at")
    var lastDownloadedAt: OffsetDateTime? = null,

    @Column(name = "download_count", nullable = false)
    var downloadCount: Int = 0,
)
