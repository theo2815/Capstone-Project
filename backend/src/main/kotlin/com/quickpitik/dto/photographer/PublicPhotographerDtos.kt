package com.quickpitik.dto.photographer

// Mirrors website/src/lib/photographer-registry.ts CoverSource — tagged union:
//   { kind: "gradient"; from; to } | { kind: "image"; url }
// Jackson serializes nullable optionals as null fields; the FE narrows on `kind`.
data class CoverSourceDto(
    val kind: String,
    val from: String? = null,
    val to: String? = null,
    val url: String? = null,
)

// Mirrors website/src/lib/photographer-registry.ts PhotographerEventCoverage.
data class PhotographerEventCoverageDto(
    val eventSlug: String,
    val state: String,
    val photoCount: Int,
    val salesCount: Int,
    // V46: the portfolio hides UNLISTED cards; the share page still resolves
    // coverage from this list, which is why they are carried, not filtered.
    val visibility: String = "public",
    val pricingMode: String = "paid",
)

// Mirrors website/src/lib/photographer-registry.ts PhotographerProfile.
// displayName defaults to user.name when brand_name is unset; city is null
// until PR 8 wires regions → city derivation.
data class PhotographerProfileDto(
    val handle: String,
    val displayName: String,
    val brandColor: String,
    val bio: String,
    val city: String?,
    val memberSince: String,
    val cover: CoverSourceDto?,
    val watermarkLabel: String?,
    val events: List<PhotographerEventCoverageDto>,
)
