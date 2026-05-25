package com.quickpitik.mobile.data.remote

import okhttp3.MultipartBody
import retrofit2.http.*

interface QuickPitikApi {
    @POST("api/v1/auth/login")
    suspend fun login(@Body request: LoginRequest): ApiResponseEnvelope<AuthResponse>

    @POST("api/v1/auth/register")
    suspend fun register(@Body request: RegisterRequest): ApiResponseEnvelope<AuthResponse>

    @Multipart
    @POST("api/v1/me/photographer/events/{eventId}/photos")
    suspend fun uploadPhoto(
        @Header("Authorization") token: String,
        @Path("eventId") eventId: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<UploadedPhotoDto>

    @GET("api/v1/me/photographer/events")
    suspend fun getPhotographerEvents(
        @Header("Authorization") token: String,
        @Query("withUploads") withUploads: Boolean = false
    ): ApiResponseEnvelope<PaginatedResponse<PhotographerEventSummaryDto>>

    @GET("api/v1/me/photographer/events/{eventId}/photos")
    suspend fun getPhotographerEventPhotos(
        @Header("Authorization") token: String,
        @Path("eventId") eventId: String,
        @Query("offset") offset: Int = 0,
        @Query("limit") limit: Int = 100
    ): ApiResponseEnvelope<PaginatedResponse<PhotographerLibraryPhotoDto>>

    @GET("api/v1/me/photographer/verification")
    suspend fun getVerificationStatus(
        @Header("Authorization") token: String
    ): ApiResponseEnvelope<VerificationSubmitResponseDto>

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

    @Multipart
    @POST("api/v1/me/photographer/watermark")
    suspend fun uploadWatermark(
        @Header("Authorization") token: String,
        @Part file: MultipartBody.Part
    ): ApiResponseEnvelope<MediaUploadResponseDto>

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

    @GET("api/v1/events/{slug}/photos")
    suspend fun getEventPhotos(
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
