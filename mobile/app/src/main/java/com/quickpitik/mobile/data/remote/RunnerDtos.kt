package com.quickpitik.mobile.data.remote

data class EventDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String, // "YYYY-MM-DD"
    val location: String,
    val bannerUrl: String?,
    val photoCount: Int,
    val participantCount: Int,
    val status: String, // "LIVE", "COMPLETED", etc.
    // Photographer-owned events (V46). Only PUBLIC events reach the list, so
    // visibility is informational; pricingMode == "free" tells a tile it's free.
    // Wire is lowercase: visibility public|unlisted, pricingMode paid|free.
    val visibility: String = "public",
    val pricingMode: String = "paid",
)

// GET /events/{slug} — backend dto/events/EventDetailDto. The list DTO above
// deliberately omits the editorial fields (description, organizer, categories,
// pricing); this carries them for the cockpit's AboutStrip. All content
// fields nullable-with-defaults so a trimmed backend payload can't NPE Gson.
data class EventDetailDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val location: String,
    val bannerUrl: String? = null,
    val photoCount: Int = 0,
    val participantCount: Int = 0,
    val status: String = "",
    val description: String? = null,
    val organizerName: String? = null,
    val categories: List<String> = emptyList(),
    val pricePerPhoto: Double = 0.0,
    val bundlePrice: Double? = null,
    val bundleSize: Int? = null,
    // V46 owned/free events. pricingMode == "free" drives the runner free-download
    // flow; photographerHandle credits the owner ("Free from @handle"). Lowercase
    // wire: visibility public|unlisted, pricingMode paid|free, watermarkPolicy
    // platform|own|none.
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val watermarkPolicy: String = "platform",
    val photographerHandle: String? = null,
)

data class PhotoDto(
    val id: String,
    val bib: String?,
    val km: Int?,
    val time: String, // "HH:MM"
    val tone: Int,
    val span: String, // e.g. "portrait"
    val price: Double,
    val imageUrl: String?,
    val alt: String?,
    // Presigned URL for the un-watermarked original. Backend populates it only
    // for photos this user has actually bought (see backend PhotoDto + G-2);
    // null for everyone else. The lightbox prefers it so a runner stops seeing
    // the watermark on a photo they paid for. Grid thumbnails stay watermarked,
    // matching the website.
    val cleanUrl: String? = null,
    // Who took the shot, so the runner can tap through to that photographer's
    // public profile. Both come from backend PhotoDto (2026-08-15).
    //
    // A null handle is a CONTRACT, not a missing value: PhotographerSettings.handle
    // is only assigned at verification, so an unverified photographer has a name
    // and no handle. Render the name as plain text there — never a tap target,
    // or the runner lands on /{null}. Both are null on legacy/seed rows that
    // carry no photographerId at all.
    val photographerHandle: String? = null,
    val photographerName: String? = null,
    // Photographer coupon (V45/V47): set only when the photographer has a live
    // coupon AND the photo has a price. couponPrice is what the runner pays with
    // the code (list price − discount), computed server-side — mobile never does
    // money math. Backend PhotoDto has NO eventId (the vault parity note said
    // otherwise; code wins) — event context comes from the screen, not the photo.
    val couponCode: String? = null,
    val couponPercentOff: Int? = null,
    val couponPrice: Double? = null,
    // Free event (V46): preview is unmarked and the original is anyone's.
    // downloadUrl is an attachment URL for the clean file (cleanUrl is also set
    // for every visitor); both null on paid events unless owned.
    val free: Boolean = false,
    val downloadUrl: String? = null,
    val metadataAccess: String? = null,
    // Small watermarked tile for photo grids (backend FS-9). Null on photos
    // published before it existed; grids then fall back to imageUrl. The
    // lightbox always uses imageUrl / cleanUrl.
    val gridUrl: String? = null,
)

data class SearchByFaceJsonRequest(
    val selfieId: String? = null,
    // True = the backend matches with every selfie in the library (union).
    val allSelfies: Boolean? = null,
    val offset: Int = 0,
    val limit: Int = 100
)

// F8 (2026-05-27): Mirror of backend dto/runner/RunnerMessageDto. Same shape
// as PhotographerMessageDto + an `orderId` for deep-linking from a dispute
// outcome notification back to the /orders detail page on the website
// (mobile reuses the same field to mark the related order in its inbox UI).
data class RunnerMessageDto(
    val id: String,
    val kind: String,           // snake_case wire form, see backend RunnerMessageKind
    val title: String?,
    val body: String,
    val orderId: String?,
    val sourceDecisionId: String?,
    val createdAt: String,
    val readAt: String?,
)

// Push frame on /ws/me/runner/notifications. The backend builds `message` from
// the same field set as the REST DTO above (RunnerMessagesService.pushMessage),
// so it deserializes straight into RunnerMessageDto with nothing missing.
data class RunnerMessageFrame(
    val type: String?,
    val message: RunnerMessageDto?,
)

// POST /public/photos/verify — "Whose photo is this?" reverse attribution (V42).
// Attribution only; never a photo id/URL. confidence wire: "strong" | "weak" |
// null. Backend PhotoVerifyResultDto.
data class PhotoVerifyResultDto(
    val matched: Boolean,
    val confidence: String? = null,
    val photographerName: String? = null,
    val photographerHandle: String? = null,
    val eventName: String? = null,
    val eventDate: String? = null,
)
