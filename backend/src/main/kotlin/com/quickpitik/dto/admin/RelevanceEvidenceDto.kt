package com.quickpitik.dto.admin

import java.util.UUID

// PE-2.3: GET /api/v1/admin/flags/{flagId}/relevance. Admins only: a copy sample names the other
// photographer (C6); nothing here is ever sent to a photographer (D18).
data class RelevanceEvidenceDto(
    val flagId: UUID,
    val payoutHoldAt: String?,
    val batches: List<RelevanceBatchDto>,
)

// One batch (event x photographer) linked to the flag. The counts are the last scoring's; null means
// not scored or, for a photographer's own event, no date rule (C8).
data class RelevanceBatchDto(
    val eventId: UUID,
    val eventName: String,
    val eventDate: String,
    val sharedRace: Boolean,
    val level: String?,
    val copyLow: Boolean?,
    val dateLow: Boolean?,
    val photosLive: Int,
    val lookups: Int,
    val tightHits: Int,
    val weakHits: Int,
    val exactMatches: Int,
    val xmpMatches: Int,
    val datedPhotos: Int?,
    val datedOutside: Int?,
    val metadataPending: Int?,
    val bibbedPhotos: Int?,
    val bibOverlapPhotos: Int?,
    val scoredAt: String?,
    val copySamples: List<CopySampleDto>,
    val dateSamples: List<DateSampleDto>,
    // Photos this flag hid in this event and an admin has not restored (PE-2.4, R2-07).
    val hiddenPhotos: Int = 0,
)

// match* is the nearest photo by another photographer in the event; exact* the earliest same file by
// another photographer in any event. A gone photo (missing row or trashed) has no thumbnail; a
// photographer handle is shown only while the photo still has an owner (R2-15).
data class CopySampleDto(
    val photoId: UUID,
    val thumbnailUrl: String?,
    val matchPhotoId: UUID?,
    val matchThumbnailUrl: String?,
    val matchPhotographerHandle: String?,
    val distance: Int?,
    val pair: String?,
    val exact: Boolean,
    val exactPhotoId: UUID? = null,
    val exactThumbnailUrl: String? = null,
    val exactPhotographerHandle: String? = null,
    val exactEventId: UUID? = null,
)

// A batch photo whose capture date (in Manila) is outside the event date window.
data class DateSampleDto(
    val photoId: UUID,
    val thumbnailUrl: String?,
    val takenOn: String,
)
