package com.quickpitik.mobile.data.remote

import okhttp3.MultipartBody
import retrofit2.Call
import retrofit2.http.*

interface QuickPitikApi {
    @POST("api/v1/auth/login")
    suspend fun login(@Body request: LoginRequest): ApiResponseEnvelope<AuthResponse>

    @POST("api/v1/auth/register")
    suspend fun register(@Body request: RegisterRequest): ApiResponseEnvelope<AuthResponse>

    // Non-suspend Call on purpose: TokenAuthenticator runs on OkHttp's thread
    // outside any coroutine, so it needs a blocking execute(). Called only via
    // RetrofitClient.refreshApi (the authenticator-free client).
    @POST("api/v1/auth/refresh")
    fun refreshToken(@Body request: RefreshRequest): Call<ApiResponseEnvelope<AuthResponse>>

    // Best-effort refresh-token revocation on sign-out, matching the website
    // (`use-auth.ts` logout). Public, so it carries no Authorization header.
    // Callers must not block sign-out on the result — see AuthViewModel.logout.
    @POST("api/v1/auth/logout")
    suspend fun logout(
        @Body request: LogoutRequest
    ): ApiResponseEnvelope<Map<String, Boolean>>

    // Auth recovery. Both are public (SecurityConfig permits /auth/**) so they
    // carry no Authorization header, and TokenAuthenticator skips /auth/* — a
    // 4xx here can never trigger a refresh or a forced logout.
    @POST("api/v1/auth/forgot-password")
    suspend fun forgotPassword(
        @Body request: ForgotPasswordRequest
    ): ApiResponseEnvelope<MessageResponse>

    @POST("api/v1/auth/verify-reset-otp")
    suspend fun verifyResetOtp(
        @Body request: VerifyResetOtpRequest
    ): ApiResponseEnvelope<VerifyResetOtpResponse>

    @POST("api/v1/auth/reset-password")
    suspend fun resetPassword(
        @Body request: ResetPasswordRequest
    ): ApiResponseEnvelope<MessageResponse>

    @Multipart
    @POST("api/v1/me/photographer/events/{eventId}/photos")
    suspend fun uploadPhoto(
        @Header("Authorization") token: String,
        @Path("eventId") eventId: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<UploadedPhotoDto>

    // Dedup pre-flight — see PhotoExistsRequest. Read-only; the cost (and the
    // rate limit) lives on the upload this is trying to avoid.
    @POST("api/v1/me/photographer/events/{eventId}/photos/exists")
    suspend fun checkPhotosExist(
        @Header("Authorization") token: String,
        @Path("eventId") eventId: String,
        @Body request: PhotoExistsRequest
    ): ApiResponseEnvelope<PhotoExistsResponse>

    // limit defaults to the backend's MAX_LIMIT (200): without it the server
    // default of 50 silently truncated a busy photographer's covered events —
    // no caller pages through this list.
    @GET("api/v1/me/photographer/events")
    suspend fun getPhotographerEvents(
        @Header("Authorization") token: String,
        @Query("withUploads") withUploads: Boolean = false,
        @Query("limit") limit: Int = 200
    ): ApiResponseEnvelope<PaginatedResponse<PhotographerEventSummaryDto>>

    @GET("api/v1/me/photographer/events/{eventId}/photos")
    suspend fun getPhotographerEventPhotos(
        @Header("Authorization") token: String,
        @Path("eventId") eventId: String,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<PhotographerLibraryPhotoDto>>

    // Clean original for a photo the caller uploaded. The library listing only
    // carries the watermarked thumbnailUrl, so the share page resolves this on
    // demand when the photographer taps Download in the lightbox.
    @GET("api/v1/me/photographer/photos/{photoId}/download")
    suspend fun getPhotographerPhotoDownload(
        @Header("Authorization") token: String,
        @Path("photoId") photoId: String
    ): ApiResponseEnvelope<PhotographerDownloadDto>

    @GET("api/v1/me/photographer/verification")
    suspend fun getVerificationStatus(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<VerificationSubmitResponseDto>

    // Public reference data — no Authorization header. Backend owns the list;
    // see RegionDto for why neither client hardcodes it any more.
    @GET("api/v1/regions")
    suspend fun getRegions(): ApiResponseEnvelope<List<RegionDto>>

    @GET("api/v1/me/photographer/brand")
    suspend fun getBrandSettings(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<BrandSettingsResponseDto>

    @PUT("api/v1/me/photographer/brand")
    suspend fun updateBrand(
        @Header("Authorization") token: String,
        @Body request: BrandPatchRequest
    ): ApiResponseEnvelope<Map<String, Any?>>

    @PUT("api/v1/me/photographer/handle")
    suspend fun updateHandle(
        @Header("Authorization") token: String,
        @Body request: HandlePatchRequest
    ): ApiResponseEnvelope<Map<String, Any?>>

    @PUT("api/v1/me/photographer/region")
    suspend fun updateRegion(
        @Header("Authorization") token: String,
        @Body request: RegionPatchRequest
    ): ApiResponseEnvelope<Map<String, Any?>>

    @POST("api/v1/me/photographer/socials")
    suspend fun createSocial(
        @Header("Authorization") token: String,
        @Body request: CreateSocialRequest
    ): ApiResponseEnvelope<Map<String, Any?>>

    @GET("api/v1/me/photographer/socials")
    suspend fun getSocials(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<List<SocialLinkDto>>

    @POST("api/v1/me/photographer/payouts")
    suspend fun createPayoutAccount(
        @Header("Authorization") token: String,
        @Body request: CreatePayoutRequest
    ): ApiResponseEnvelope<PayoutAccountDto>

    @PATCH("api/v1/me/photographer/socials/{id}")
    suspend fun patchSocial(
        @Header("Authorization") token: String,
        @Path("id") id: String,
        @Body request: PatchSocialRequest
    ): ApiResponseEnvelope<SocialLinkDto>

    @DELETE("api/v1/me/photographer/socials/{id}")
    suspend fun deleteSocial(
        @Header("Authorization") token: String,
        @Path("id") id: String
    ): ApiResponseEnvelope<RemovedResponse>

    @PATCH("api/v1/me/photographer/payouts/{id}")
    suspend fun patchPayout(
        @Header("Authorization") token: String,
        @Path("id") id: String,
        @Body request: PatchPayoutRequest
    ): ApiResponseEnvelope<PayoutAccountDto>

    @DELETE("api/v1/me/photographer/payouts/{id}")
    suspend fun deletePayout(
        @Header("Authorization") token: String,
        @Path("id") id: String
    ): ApiResponseEnvelope<RemovedResponse>

    @PATCH("api/v1/me/photographer/payouts/{id}/primary")
    suspend fun setPrimaryPayout(
        @Header("Authorization") token: String,
        @Path("id") id: String
    ): ApiResponseEnvelope<List<PayoutAccountDto>>

    @Multipart
    @POST("api/v1/me/photographer/payouts/{id}/qr")
    suspend fun uploadPayoutQr(
        @Header("Authorization") token: String,
        @Path("id") id: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<PayoutAccountDto>

    @Multipart
    @POST("api/v1/me/photographer/watermark")
    suspend fun uploadWatermark(
        @Header("Authorization") token: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<MediaUploadResponseDto>

    // Clears the avatar and returns the updated user (avatarUrl null).
    @DELETE("api/v1/me/avatar")
    suspend fun deleteAvatar(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<UserDto>

    @Multipart
    @POST("api/v1/me/avatar")
    suspend fun uploadAvatar(
        @Header("Authorization") token: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<UserDto>

    @Multipart
    @POST("api/v1/me/photographer/cover")
    suspend fun uploadCover(
        @Header("Authorization") token: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<MediaUploadResponseDto>

    @POST("api/v1/me/photographer/verification")
    suspend fun submitVerification(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<VerificationSubmitResponseDto>

    @POST("api/v1/me/photographer/verification/withdraw")
    suspend fun withdrawVerification(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<VerificationSubmitResponseDto>


    @GET("api/v1/me/photographer/earnings")
    suspend fun getEarningsOverview(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<EarningsOverviewDto>

    @GET("api/v1/me/photographer/payouts/balance")
    suspend fun getPayoutBalance(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<PayoutBalanceDto>

    @GET("api/v1/me/photographer/payouts")
    suspend fun getPayoutAccounts(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<List<PayoutAccountDto>>

    @POST("api/v1/me/photographer/payouts/request")
    suspend fun requestPayout(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<PhotographerPayoutDto>

    // Photographer cancels a HELD payout request after fixing what the admin
    // flagged; the backend hard-deletes the cycle so a fresh request can be
    // filed immediately. Body is empty ({success:true, data:null}).
    @POST("api/v1/me/photographer/payouts/{id}/withdraw")
    suspend fun withdrawPayout(
        @Header("Authorization") token: String,
        @Path("id") id: String
    ): ApiResponseEnvelope<Any?>

    @GET("api/v1/me/photographer/billing/transactions")
    suspend fun getTransactionsLedger(
        @Header("Authorization") token: String,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 50
    ): ApiResponseEnvelope<TransactionsLedgerResponse>

    @GET("api/v1/public/photographers/{handle}")
    suspend fun getPublicPhotographerProfile(
        @Path("handle") handle: String
    ): ApiResponseEnvelope<PhotographerProfileDto>

    @GET("api/v1/public/photographers/{handle}/events/{slug}/photos")
    suspend fun getPublicPhotographerEventPhotos(
        @Path("handle") handle: String,
        @Path("slug") slug: String,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<PhotoDto>>

    @GET("api/v1/events")
    suspend fun getPublicEvents(
        @Query("status") status: String = "ACTIVE",
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<EventDto>>

    // Event editorial detail (organizer, description, categories, pricing) —
    // the cockpit's AboutStrip. The list endpoint deliberately omits these.
    @GET("api/v1/events/{slug}")
    suspend fun getEventDetail(
        @Path("slug") slug: String
    ): ApiResponseEnvelope<EventDetailDto>

    // The route is public, but the bearer matters when signed in: the backend
    // reads principal?.userId to populate PhotoDto.cleanUrl for photos the
    // caller owns (unwatermarked lightbox) and to rate-bucket bib search per
    // user instead of per IP (shared race-day Wi-Fi). Nullable so the header is
    // simply omitted if a token is ever absent.
    @GET("api/v1/events/{slug}/photos")
    suspend fun getEventPhotos(
        @Header("Authorization") token: String? = null,
        @Path("slug") slug: String,
        @Query("bib") bib: String? = null,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<PhotoDto>>

    @Multipart
    @POST("api/v1/events/{slug}/photos/search-by-face")
    suspend fun searchPhotosByFace(
        @Header("Authorization") token: String,
        @Path("slug") slug: String,
        @Part selfie: MultipartBody.Part,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<PhotoDto>>

    @POST("api/v1/events/{slug}/photos/search-by-face")
    suspend fun searchPhotosByFaceJson(
        @Header("Authorization") token: String,
        @Path("slug") slug: String,
        @Body request: SearchByFaceJsonRequest
    ): ApiResponseEnvelope<PaginatedResponse<PhotoDto>>

    // "Notify me when my photos are ready" opt-in (RUNNER). The backend matches
    // the runner's selfie against the event during its date-based sweep and
    // emails once when photos of them appear. See PhotoAlertDtos.
    @GET("api/v1/events/{slug}/photo-alert")
    suspend fun getPhotoAlertStatus(
        @Header("Authorization") token: String,
        @Path("slug") slug: String
    ): ApiResponseEnvelope<PhotoAlertStatusDto>

    @POST("api/v1/events/{slug}/photo-alert")
    suspend fun registerPhotoAlert(
        @Header("Authorization") token: String,
        @Path("slug") slug: String,
        @Body request: PhotoAlertRequest
    ): ApiResponseEnvelope<PhotoAlertStatusDto>

    @DELETE("api/v1/events/{slug}/photo-alert")
    suspend fun unregisterPhotoAlert(
        @Header("Authorization") token: String,
        @Path("slug") slug: String
    ): ApiResponseEnvelope<RemovedResponse>

    @GET("api/v1/me/cart")
    suspend fun getCart(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<List<CartItemDto>>

    @POST("api/v1/me/cart/items")
    suspend fun addCartItem(
        @Header("Authorization") token: String,
        @Body request: AddCartItemRequest
    ): ApiResponseEnvelope<CartItemDto>

    @DELETE("api/v1/me/cart/items/{photoId}")
    suspend fun removeCartItem(
        @Header("Authorization") token: String,
        @Path("photoId") photoId: String
    ): ApiResponseEnvelope<RemovedResponse>

    @POST("api/v1/me/cart/merge")
    suspend fun mergeCart(
        @Header("Authorization") token: String,
        @Body request: MergeCartRequest
    ): ApiResponseEnvelope<List<CartItemDto>>

    @DELETE("api/v1/me/cart")
    suspend fun clearCart(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<ClearedResponse>

    @POST("api/v1/orders")
    suspend fun createOrder(
        @Header("Authorization") token: String?,
        @Header("Idempotency-Key") idempotencyKey: String,
        @Body request: CreateOrderRequest
    ): ApiResponseEnvelope<OrderResponse>

    @GET("api/v1/me/orders")
    suspend fun getOrders(
        @Header("Authorization") token: String,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<OrderListItemDto>>

    @GET("api/v1/me/orders/{id}")
    suspend fun getOrderDetail(
        @Header("Authorization") token: String,
        @Path("id") orderId: String
    ): ApiResponseEnvelope<OrderDetailDto>

    @POST("api/v1/me/orders/{id}/refund")
    suspend fun submitRefund(
        @Header("Authorization") token: String,
        @Path("id") orderId: String,
        @Body request: RefundRequest
    ): ApiResponseEnvelope<RefundResponse>

    @POST("api/v1/me/disputes/{id}/withdraw")
    suspend fun withdrawDispute(
        @Header("Authorization") token: String,
        @Path("id") disputeId: String
    ): ApiResponseEnvelope<RunnerDisputeDto>

    @GET("api/v1/me/photographer/messages")
    suspend fun getPhotographerMessages(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<List<PhotographerMessageDto>>

    @PATCH("api/v1/me/photographer/messages/{id}/read")
    suspend fun markMessageRead(
        @Header("Authorization") token: String,
        @Path("id") id: String
    ): ApiResponseEnvelope<PhotographerMessageDto>

    @PATCH("api/v1/me/photographer/messages/read-all")
    suspend fun markAllMessagesRead(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<MarkAllReadResponse>

    @DELETE("api/v1/me/photographer/messages/{id}")
    suspend fun removePhotographerMessage(
        @Header("Authorization") token: String,
        @Path("id") id: String
    ): ApiResponseEnvelope<MessageRemovedResponse>

    // F8 (2026-05-27): Runner inbox endpoints. Mirrors photographer messages
    // above; backend controller is MeRunnerMessagesController (PreAuthorize
    // hasRole RUNNER). DTOs share MarkAllReadResponse + MessageRemovedResponse
    // with the photographer flow.
    @GET("api/v1/me/runner/messages")
    suspend fun getRunnerMessages(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<List<RunnerMessageDto>>

    @PATCH("api/v1/me/runner/messages/{id}/read")
    suspend fun markRunnerMessageRead(
        @Header("Authorization") token: String,
        @Path("id") id: String
    ): ApiResponseEnvelope<RunnerMessageDto>

    @PATCH("api/v1/me/runner/messages/read-all")
    suspend fun markAllRunnerMessagesRead(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<MarkAllReadResponse>

    @DELETE("api/v1/me/runner/messages/{id}")
    suspend fun removeRunnerMessage(
        @Header("Authorization") token: String,
        @Path("id") id: String
    ): ApiResponseEnvelope<MessageRemovedResponse>

    @GET("api/v1/me/selfies")
    suspend fun getSelfies(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<List<SelfieRefDto>>

    @Multipart
    @POST("api/v1/me/selfies")
    suspend fun uploadSelfie(
        @Header("Authorization") token: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<SelfieRefDto>

    @DELETE("api/v1/me/selfies/{id}")
    suspend fun deleteSelfie(
        @Header("Authorization") token: String,
        @Path("id") selfieId: String
    ): ApiResponseEnvelope<RemovedResponse>

    @POST("api/v1/me/selfies/{id}/set-primary")
    suspend fun setPrimarySelfie(
        @Header("Authorization") token: String,
        @Path("id") selfieId: String
    ): ApiResponseEnvelope<List<SelfieRefDto>>

    @PUT("api/v1/me/profile")
    suspend fun updateProfile(
        @Header("Authorization") token: String,
        @Body request: ProfileUpdateRequest
    ): ApiResponseEnvelope<UserDto>

    @PUT("api/v1/me/password")
    suspend fun changePassword(
        @Header("Authorization") token: String,
        @Body request: PasswordChangeRequest
    ): ApiResponseEnvelope<Map<String, String>>

    // Step 1 of 2 — mails a confirmation link to the NEW address and changes
    // nothing yet. Step 2 (`POST /auth/confirm-email-change`) is web-only; the
    // backend links the mail to the website origin. See EmailChangeRequest.
    @PUT("api/v1/me/email")
    suspend fun requestEmailChange(
        @Header("Authorization") token: String,
        @Body request: EmailChangeRequest
    ): ApiResponseEnvelope<Map<String, String>>

    @GET("api/v1/me/saved-events")
    suspend fun getSavedEvents(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<List<SavedEventSummaryDto>>

    @POST("api/v1/me/saved-events")
    suspend fun saveEvent(
        @Header("Authorization") token: String,
        @Body request: SaveEventRequest
    ): ApiResponseEnvelope<SavedEventSummaryDto>

    @DELETE("api/v1/me/saved-events/{eventId}")
    suspend fun unsaveEvent(
        @Header("Authorization") token: String,
        @Path("eventId") eventId: String
    ): ApiResponseEnvelope<RemovedResponse>
}
