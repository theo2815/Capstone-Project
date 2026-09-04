package com.quickpitik.controller

import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.orders.CouponDto
import com.quickpitik.dto.orders.UpsertCouponRequest
import com.quickpitik.dto.photographer.BrandPatchRequest
import com.quickpitik.dto.photographer.CreatePayoutRequest
import com.quickpitik.dto.photographer.CreateSocialRequest
import com.quickpitik.dto.photographer.HandlePatchRequest
import com.quickpitik.dto.photographer.MediaUploadResponseDto
import com.quickpitik.dto.photographer.PatchPayoutRequest
import com.quickpitik.dto.photographer.PatchSocialRequest
import com.quickpitik.dto.photographer.PayoutAccountDto
import com.quickpitik.dto.photographer.RegionPatchRequest
import com.quickpitik.dto.photographer.SocialLinkDto
import com.quickpitik.dto.photographer.VerificationSubmitResponseDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.earnings.PayoutCycleService
import com.quickpitik.service.orders.CouponService
import com.quickpitik.service.photographer.PayoutAccountService
import com.quickpitik.service.photographer.PhotographerSettingsService
import com.quickpitik.service.photographer.SocialLinkService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.ratelimit.acquireOrThrow
import jakarta.validation.Valid
import org.springframework.http.MediaType
import org.springframework.security.access.prepost.PreAuthorize
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PatchMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.PutMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

/**
 * Phase F.2 photographer-settings surface. All endpoints require role
 * PHOTOGRAPHER (enforced via class-level @PreAuthorize). Lazy-creates the
 * photographer_settings row on first touch via PhotographerSettingsService.
 *
 * The /payouts GET dispatches by query-param presence per Q-009: no params
 * returns the photographer's PayoutAccount[] (settings page); with offset/limit
 * returns PaginatedResponse<PhotographerPayout> (earnings cycles, PR 9). PR 8
 * ships an empty-paginated stub for the cycle branch so the dispatcher
 * signature is locked before PR 9 fills it in.
 */
@RestController
@RequestMapping("/api/v1/me/photographer")
@PreAuthorize("hasRole('PHOTOGRAPHER')")
class MePhotographerSettingsController(
    private val photographerSettingsService: PhotographerSettingsService,
    private val socialLinkService: SocialLinkService,
    private val payoutAccountService: PayoutAccountService,
    private val payoutCycleService: PayoutCycleService,
    private val rateLimiter: RateLimiter,
    private val couponService: CouponService,
) {
    // ─── Brand / Handle / Region ──────────────────────────────────────────
    @GetMapping("/brand")
    fun getBrand(
        @AuthenticationPrincipal principal: AuthPrincipal,
    ): Map<String, Any?> = photographerSettingsService.getBrandDetails(principal.userId)

    @PutMapping("/brand")
    fun putBrand(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: BrandPatchRequest,
    ): Map<String, Any?> {
        photographerSettingsService.putBrand(principal.userId, body)
        // FE wiring uses `api.put<unknown>` and ignores body — return an empty
        // object so the envelope's `data` is not null (matches Avatar/Selfie
        // pattern of returning the canonical object after the write).
        return emptyMap()
    }

    @PutMapping("/handle")
    fun putHandle(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: HandlePatchRequest,
    ): Map<String, Any?> {
        val saved = photographerSettingsService.putHandle(principal.userId, body)
        return mapOf("handle" to saved.handle)
    }

    @PutMapping("/region")
    fun putRegion(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: RegionPatchRequest,
    ): Map<String, Any?> {
        val saved = photographerSettingsService.putRegion(principal.userId, body)
        return mapOf(
            "regionCode" to saved.regionCode,
            "provinceCode" to saved.provinceCode,
            "city" to saved.city,
        )
    }

    // ─── Cover / Watermark uploads ────────────────────────────────────────
    @PostMapping("/cover", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun uploadCover(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestPart("file") file: MultipartFile,
    ): MediaUploadResponseDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, principal.userId.toString())
        return photographerSettingsService.uploadCover(
            userId = principal.userId,
            bytes = file.bytes,
            contentType = file.contentType,
        )
    }

    @PostMapping("/watermark", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun uploadWatermark(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestPart("file") file: MultipartFile,
    ): MediaUploadResponseDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, principal.userId.toString())
        return photographerSettingsService.uploadWatermark(
            userId = principal.userId,
            bytes = file.bytes,
            contentType = file.contentType,
        )
    }

    // ─── Socials CRUD ─────────────────────────────────────────────────────
    @GetMapping("/socials")
    fun listSocials(@AuthenticationPrincipal principal: AuthPrincipal): List<SocialLinkDto> =
        socialLinkService.list(principal.userId)

    @PostMapping("/socials")
    fun createSocial(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: CreateSocialRequest,
    ): SocialLinkDto = socialLinkService.create(principal.userId, body)

    @PatchMapping("/socials/{id}")
    fun patchSocial(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
        @Valid @RequestBody body: PatchSocialRequest,
    ): SocialLinkDto = socialLinkService.patch(principal.userId, id, body)

    @DeleteMapping("/socials/{id}")
    fun deleteSocial(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): Map<String, Boolean> {
        socialLinkService.delete(principal.userId, id)
        return mapOf("removed" to true)
    }

    // ─── Event coupon ─────────────────────────────────────────────────────
    @GetMapping("/events/{eventId}/coupon")
    fun getCoupon(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): CouponDto? = couponService.get(principal.userId, eventId)

    @PutMapping("/events/{eventId}/coupon")
    fun putCoupon(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
        @Valid @RequestBody body: UpsertCouponRequest,
    ): CouponDto = couponService.upsert(principal.userId, eventId, body)

    @DeleteMapping("/events/{eventId}/coupon")
    fun deleteCoupon(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable eventId: UUID,
    ): Map<String, Boolean> {
        couponService.delete(principal.userId, eventId)
        return mapOf("removed" to true)
    }

    // ─── Payouts CRUD + primary toggle + QR ───────────────────────────────
    // Path-overlap dispatch per Q-009: no params → PayoutAccount[] (settings);
    // offset/limit → PaginatedResponse<PhotographerPayout> (earnings cycles,
    // PR 9). One handler so the dispatcher signature stays locked at the
    // controller surface — the cycle branch resolves to PayoutCycleService.
    @GetMapping("/payouts")
    fun listPayouts(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestParam(required = false) offset: Int?,
        @RequestParam(required = false) limit: Int?,
    ): Any = if (offset != null || limit != null) {
        payoutCycleService.list(
            photographerId = principal.userId,
            params = PaginationParams.of(offset, limit),
        )
    } else {
        payoutAccountService.list(principal.userId)
    }

    @PostMapping("/payouts")
    fun createPayout(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @Valid @RequestBody body: CreatePayoutRequest,
    ): PayoutAccountDto = payoutAccountService.create(principal.userId, body)

    @PatchMapping("/payouts/{id}")
    fun patchPayout(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
        @Valid @RequestBody body: PatchPayoutRequest,
    ): PayoutAccountDto = payoutAccountService.patch(principal.userId, id, body)

    @DeleteMapping("/payouts/{id}")
    fun deletePayout(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): Map<String, Boolean> {
        payoutAccountService.delete(principal.userId, id)
        return mapOf("removed" to true)
    }

    @PatchMapping("/payouts/{id}/primary")
    fun setPrimaryPayout(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): List<PayoutAccountDto> = payoutAccountService.setPrimary(principal.userId, id)

    @PostMapping("/payouts/{id}/qr", consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun uploadPayoutQr(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
        @RequestPart("file") file: MultipartFile,
    ): PayoutAccountDto {
        rateLimiter.acquireOrThrow(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, principal.userId.toString())
        return payoutAccountService.uploadQr(
            userId = principal.userId,
            id = id,
            bytes = file.bytes,
            contentType = file.contentType,
        )
    }

    // ─── Verification submit + status read ────────────────────────────────
    @GetMapping("/verification")
    fun getVerification(@AuthenticationPrincipal principal: AuthPrincipal): VerificationSubmitResponseDto =
        photographerSettingsService.getVerificationStatus(principal.userId)

    @PostMapping("/verification")
    fun submitVerification(@AuthenticationPrincipal principal: AuthPrincipal): VerificationSubmitResponseDto =
        photographerSettingsService.submitVerification(principal.userId)

    @PostMapping("/verification/withdraw")
    fun withdrawVerification(@AuthenticationPrincipal principal: AuthPrincipal): VerificationSubmitResponseDto =
        photographerSettingsService.withdrawVerification(principal.userId)
}
