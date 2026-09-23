package com.quickpitik.mobile.ui.photographer

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.readAtMost
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * What the photographer's last action is doing, and what to tell them about it.
 *
 * Replaces the `String?` channel whose `"Success: "` / `"Error: "` prefixes the
 * settings screen had to parse (`startsWith("Success")`) to decide whether a
 * save had actually landed. The outcome is the type now; [message] is the
 * sentence, with no prefix to re-read.
 */
sealed interface ActionState {
    /** The sentence to show, or null when there is nothing to say. */
    val message: String?

    object Idle : ActionState {
        override val message: String? = null
    }

    /** In flight — the copy the old channel showed while the caller waited. */
    data class Busy(override val message: String) : ActionState
    data class Done(override val message: String) : ActionState
    data class Failed(override val message: String) : ActionState
}

/**
 * The studio's own settings: brand, handle, region, socials, payout accounts
 * and the watermark — everything the Settings tab reads and writes, plus the
 * brand payload the shell's avatar and the profile preview need.
 *
 * Split out of [PhotographerDashboardViewModel] (plan 2 phase G). Its `init`
 * loads only the settings routes, so a test of one payout rule stubs those
 * four endpoints instead of the dashboard's whole cold-start salvo.
 *
 * Scoped to the `studio` nav graph like the dashboard ViewModel beside it, so
 * both live and die with the studio session.
 */
class StudioSettingsViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)

    private val _isFetchingBrandSettings = MutableStateFlow(false)
    val isFetchingBrandSettings: StateFlow<Boolean> = _isFetchingBrandSettings

    private val _brandSettings = MutableStateFlow<com.quickpitik.mobile.data.remote.BrandSettingsResponseDto?>(null)
    val brandSettings: StateFlow<com.quickpitik.mobile.data.remote.BrandSettingsResponseDto?> = _brandSettings

    // Non-null when the settings hydration failed with nothing cached — the
    // settings screen must not offer an empty editable form (see fetchSettings).
    private val _settingsLoadError = MutableStateFlow<String?>(null)
    val settingsLoadError: StateFlow<String?> = _settingsLoadError

    // Backend-owned PH region/province list (GET /regions). Empty until the
    // fetch lands; the region slab shows "Loading regions…" meanwhile rather
    // than an empty picker. A failed fetch leaves it empty and the photographer
    // keeps whatever region is already saved — nothing is silently cleared.
    private val _regions = MutableStateFlow<List<com.quickpitik.mobile.data.remote.RegionDto>>(emptyList())
    val regions: StateFlow<List<com.quickpitik.mobile.data.remote.RegionDto>> = _regions

    private val _payoutAccounts = MutableStateFlow<List<com.quickpitik.mobile.data.remote.PayoutAccountDto>>(emptyList())
    val payoutAccounts: StateFlow<List<com.quickpitik.mobile.data.remote.PayoutAccountDto>> = _payoutAccounts

    private val _socials = MutableStateFlow<List<com.quickpitik.mobile.data.remote.SocialLinkDto>>(emptyList())
    val socials: StateFlow<List<com.quickpitik.mobile.data.remote.SocialLinkDto>> = _socials

    private val _actionState = MutableStateFlow<ActionState>(ActionState.Idle)
    val actionState: StateFlow<ActionState> = _actionState

    private val _isSavingSettings = MutableStateFlow(false)
    val isSavingSettings: StateFlow<Boolean> = _isSavingSettings

    init {
        // Load brand + socials / payouts / regions at startup, so the settings
        // editor shows what the photographer already has (these were previously
        // loaded only on the settings-tab tap, leaving the editor briefly
        // empty). fetchSettings already loads brand, so there's no separate
        // fetchBrandSettings() call here — it would double-GET brand at startup.
        fetchSettings()
    }

    fun clearActionState() {
        _actionState.value = ActionState.Idle
    }

    fun fetchBrandSettings() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            loadBrandSettings(token)
        }
    }

    private suspend fun loadBrandSettings(token: String) {
        _isFetchingBrandSettings.value = true
        try {
            val response = RetrofitClient.apiService.getBrandSettings("Bearer $token")
            if (response.success && response.data != null) {
                _brandSettings.value = response.data
                _settingsLoadError.value = null
            }
        } catch (e: Exception) {
            // The brand payload hydrates the whole settings form. With nothing
            // cached, an empty editable form could overwrite server values.
            if (_brandSettings.value == null) {
                _settingsLoadError.value = RetrofitClient.parseError(e)
            }
        } finally {
            _isFetchingBrandSettings.value = false
        }
    }

    fun fetchSettings() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            coroutineScope {
                launch { loadBrandSettings(token) }
                launch {
                    try {
                        val response = RetrofitClient.apiService.getPayoutAccounts("Bearer $token")
                        if (response.success && response.data != null) {
                            _payoutAccounts.value = response.data
                        }
                    } catch (_: Exception) {
                        // Keep the last known list.
                    }
                }
                launch {
                    try {
                        val response = RetrofitClient.apiService.getSocials("Bearer $token")
                        if (response.success && response.data != null) {
                            _socials.value = response.data
                        }
                    } catch (_: Exception) {
                        // Keep the last known list.
                    }
                }
                // Reference data changes rarely and is cached by the backend.
                if (_regions.value.isEmpty()) launch {
                    try {
                        val response = RetrofitClient.apiService.getRegions()
                        if (response.success && response.data != null) {
                            _regions.value = response.data
                        }
                    } catch (_: Exception) {
                        // Keep the saved region visible.
                    }
                }
            }
        }
    }

    private suspend fun readSettingsAsset(uri: String): ByteArray? =
        withContext(Dispatchers.IO) {
            getApplication<Application>().contentResolver.openInputStream(Uri.parse(uri))
                ?.use { it.readAtMost(MAX_UPLOAD_BYTES + 1) }
        }

    // Returns the validated bytes, or null after setting the error state.
    private suspend fun readValidatedAsset(uri: String, label: String): ByteArray? {
        val bytes = readSettingsAsset(uri)
        if (bytes == null || bytes.isEmpty() || bytes.size > MAX_UPLOAD_BYTES) {
            _actionState.value = ActionState.Failed("$label must be a readable image up to 8 MB.")
            return null
        }
        return bytes
    }

    /**
     * [onSaved] runs after the whole chain lands. Saving brand / handle /
     * region / payout can complete the verification checklist, and the
     * verification status belongs to [PhotographerDashboardViewModel] — the
     * settings screen wires this to its `fetchVerificationStatus`.
     */
    fun saveSettings(
        brandName: String,
        bio: String,
        gcashName: String,
        gcashNumber: String,
        handle: String,
        regionCode: String,
        provinceCode: String,
        socialUrl: String,
        avatarUri: String?,
        coverUri: String?,
        watermarkUri: String?,
        // Wire value from ALLOWED_BRAND_COLORS (none/fresh/amber/indigo/rose/
        // ink). Null = leave whatever is on the server untouched.
        brandColor: String? = null,
        // Placement of the photographer's own logo (V51). Null = leave unchanged;
        // otherwise PUT it only when it differs from the loaded value.
        watermarkPlacement: com.quickpitik.mobile.data.remote.WatermarkSettingsDto? = null,
        onSaved: () -> Unit = {},
    ) {
        viewModelScope.launch {
            _isSavingSettings.value = true
            _actionState.value = ActionState.Busy("Saving settings...")
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _actionState.value = ActionState.Failed("No valid session. Please log in again.")
                _isSavingSettings.value = false
                return@launch
            }

            try {
                // 0. Read + validate every picked image BEFORE the first PATCH
                // — an unreadable or oversized pick must not abort a
                // half-applied save.
                val avatarBytes = avatarUri?.let { readValidatedAsset(it, "Avatar") ?: return@launch }
                val coverBytes = coverUri?.let { readValidatedAsset(it, "Cover") ?: return@launch }
                val watermarkBytes = watermarkUri?.let { readValidatedAsset(it, "Watermark") ?: return@launch }

                // 1. Update Brand (Name & Bio). brandColor passes through the
                // hydrated value — sending a literal "none" here silently reset
                // a brand colour the photographer had picked on the website on
                // every mobile save. (A mobile picker is separate work; until
                // then mobile must at least not destroy the setting.)
                val brandResponse = RetrofitClient.apiService.updateBrand(
                    "Bearer $token",
                    com.quickpitik.mobile.data.remote.BrandPatchRequest(
                        brandName = brandName,
                        brandColor = brandColor ?: brandSettings.value?.brandColor ?: "none",
                        bio = bio
                    )
                )
                if (!brandResponse.success) {
                    _actionState.value = ActionState.Failed(brandResponse.error ?: "Failed to update brand.")
                    _isSavingSettings.value = false
                    return@launch
                }

                // 1.1 Update Handle if configured
                if (handle.isNotBlank()) {
                    val handleResponse = RetrofitClient.apiService.updateHandle(
                        "Bearer $token",
                        com.quickpitik.mobile.data.remote.HandlePatchRequest(handle = handle)
                    )
                    if (!handleResponse.success) {
                        _actionState.value = ActionState.Failed(handleResponse.error ?: "Failed to update handle.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 1.2 Update Region if configured
                if (regionCode.isNotBlank() && provinceCode.isNotBlank()) {
                    val regionResponse = RetrofitClient.apiService.updateRegion(
                        "Bearer $token",
                        com.quickpitik.mobile.data.remote.RegionPatchRequest(
                            regionCode = regionCode,
                            provinceCode = provinceCode
                        )
                    )
                    if (!regionResponse.success) {
                        _actionState.value = ActionState.Failed(regionResponse.error ?: "Failed to update region.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 1.3 Update Social Profile link if configured
                if (socialUrl.isNotBlank()) {
                    try {
                        RetrofitClient.apiService.createSocial(
                            "Bearer $token",
                            com.quickpitik.mobile.data.remote.CreateSocialRequest(
                                platform = "facebook",
                                url = socialUrl
                            )
                        )
                    } catch (e: Exception) {
                        // Fail silently if social link already exists
                    }
                }

                // 2. Update GCash Payout — checked like every other step in
                // this chain. Ignoring the response let a rejected payout
                // account (bad number format, duplicate) report overall
                // success while the account was never saved.
                if (gcashName.isNotBlank() && gcashNumber.isNotBlank()) {
                    val payoutResponse = RetrofitClient.apiService.createPayoutAccount(
                        "Bearer $token",
                        com.quickpitik.mobile.data.remote.CreatePayoutRequest(
                            method = "gcash",
                            accountNumber = gcashNumber,
                            accountName = gcashName
                        )
                    )
                    if (!payoutResponse.success) {
                        _actionState.value =
                            ActionState.Failed(payoutResponse.error ?: "Failed to save payout account.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 3. Upload Avatar if chosen
                if (avatarBytes != null) {
                    val requestFile = avatarBytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, avatarBytes.size)
                    val part = MultipartBody.Part.createFormData("file", "avatar.jpg", requestFile)
                    val avatarResponse = RetrofitClient.apiService.uploadAvatar("Bearer $token", part)
                    if (!avatarResponse.success) {
                        _actionState.value = ActionState.Failed(avatarResponse.error ?: "Failed to upload avatar.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 4. Upload Cover if chosen
                if (coverBytes != null) {
                    val requestFile = coverBytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, coverBytes.size)
                    val part = MultipartBody.Part.createFormData("file", "cover.jpg", requestFile)
                    val coverResponse = RetrofitClient.apiService.uploadCover("Bearer $token", part)
                    if (!coverResponse.success) {
                        _actionState.value = ActionState.Failed(coverResponse.error ?: "Failed to upload cover.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 5. Upload Watermark if chosen
                if (watermarkBytes != null) {
                    val requestFile = watermarkBytes.toRequestBody("image/png".toMediaTypeOrNull(), 0, watermarkBytes.size)
                    val part = MultipartBody.Part.createFormData("file", "watermark.png", requestFile)
                    val watermarkResponse = RetrofitClient.apiService.uploadWatermark("Bearer $token", part)
                    if (!watermarkResponse.success) {
                        _actionState.value =
                            ActionState.Failed(watermarkResponse.error ?: "Failed to upload watermark.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                // 6. Update watermark placement (position/opacity/size) when the
                // photographer changed it — independent of the image upload above.
                if (watermarkPlacement != null &&
                    watermarkPlacement != brandSettings.value?.watermarkSettings
                ) {
                    val wmSettingsResponse = RetrofitClient.apiService.updateWatermarkSettings(
                        "Bearer $token",
                        watermarkPlacement,
                    )
                    if (!wmSettingsResponse.success) {
                        _actionState.value =
                            ActionState.Failed(wmSettingsResponse.error ?: "Failed to update watermark placement.")
                        _isSavingSettings.value = false
                        return@launch
                    }
                }

                _actionState.value = ActionState.Done("Settings updated successfully!")
                onSaved()
                fetchSettings()
            } catch (e: retrofit2.HttpException) {
                // The backend's envelope is {success, errors:[{code,message}]} —
                // the previous flat {"error"/"message"} parse never matched it,
                // so HANDLE_TAKEN / RESERVED_HANDLE / INVALID_REGION /
                // VALIDATION_ERROR all collapsed to a hardcoded 409 guess.
                val err = RetrofitClient.parseHttpError(e)
                _actionState.value = ActionState.Failed(err?.message
                    ?: if (e.code() == 409) "That handle is already taken by another user."
                    else (e.localizedMessage ?: "Connection error."))
            } catch (e: Exception) {
                _actionState.value = ActionState.Failed(RetrofitClient.parseError(e))
            } finally {
                _isSavingSettings.value = false
            }
        }
    }

    // ── Socials immediate-CRUD ────────────────────────────────────────────
    fun addSocialAccount(platform: String, url: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _actionState.value = ActionState.Busy("Saving social...")
            try {
                val response = RetrofitClient.apiService.createSocial(
                    "Bearer $token",
                    com.quickpitik.mobile.data.remote.CreateSocialRequest(platform, url),
                )
                if (response.success) {
                    _actionState.value = ActionState.Done("Social added.")
                    fetchSettings()
                } else {
                    _actionState.value = ActionState.Failed(response.error ?: "Failed to add social.")
                }
            } catch (e: Exception) {
                _actionState.value = ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun updateSocialAccount(id: String, url: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _actionState.value = ActionState.Busy("Updating social...")
            try {
                val response = RetrofitClient.apiService.patchSocial(
                    "Bearer $token",
                    id,
                    com.quickpitik.mobile.data.remote.PatchSocialRequest(url),
                )
                if (response.success) {
                    _actionState.value = ActionState.Done("Social updated.")
                    fetchSettings()
                } else {
                    _actionState.value = ActionState.Failed(response.error ?: "Failed to update social.")
                }
            } catch (e: Exception) {
                _actionState.value = ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun deleteSocialAccount(id: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _actionState.value = ActionState.Busy("Removing social...")
            try {
                val response = RetrofitClient.apiService.deleteSocial("Bearer $token", id)
                if (response.success) {
                    _actionState.value = ActionState.Done("Social removed.")
                    fetchSettings()
                } else {
                    _actionState.value = ActionState.Failed(response.error ?: "Failed to remove social.")
                }
            } catch (e: Exception) {
                _actionState.value = ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    // ── Payouts immediate-CRUD ────────────────────────────────────────────
    fun addPayoutAccount(
        method: String,
        accountName: String,
        accountNumber: String,
        qrUri: String? = null,
    ) {
        if (qrUri == null) {
            _actionState.value = ActionState.Failed("Upload a payout QR code before saving.")
            return
        }
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _actionState.value = ActionState.Busy("Saving payout...")
            try {
                val qrBytes = readValidatedAsset(qrUri, "QR") ?: return@launch
                val createResponse = RetrofitClient.apiService.createPayoutAccount(
                    "Bearer $token",
                    com.quickpitik.mobile.data.remote.CreatePayoutRequest(method, accountNumber, accountName),
                )
                if (!createResponse.success || createResponse.data == null) {
                    _actionState.value = ActionState.Failed(createResponse.error ?: "Failed to add payout.")
                    return@launch
                }
                val newId = createResponse.data.id
                if (!uploadPayoutQr(token, newId, qrBytes)) {
                    // Keep the incomplete account visible so Edit retries the
                    // QR upload without creating another account.
                    _actionState.value = ActionState.Failed(
                        "Payout setup is incomplete. Edit this account to upload its required QR.")
                    fetchSettings()
                    return@launch
                }
                _actionState.value = ActionState.Done("Payout added.")
                fetchSettings()
            } catch (e: Exception) {
                _actionState.value = ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun updatePayoutAccount(
        id: String,
        accountName: String,
        accountNumber: String,
        qrUri: String? = null,
    ) {
        val existing = _payoutAccounts.value.find { it.id == id }
        if (qrUri == null && (existing?.qr == null || existing.accountNumber != accountNumber)) {
            _actionState.value =
                ActionState.Failed("Upload a QR code matching this payout account before saving.")
            return
        }
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _actionState.value = ActionState.Busy("Updating payout...")
            try {
                val qrBytes = qrUri?.let { readValidatedAsset(it, "QR") ?: return@launch }
                val response = RetrofitClient.apiService.patchPayout(
                    "Bearer $token",
                    id,
                    com.quickpitik.mobile.data.remote.PatchPayoutRequest(accountNumber, accountName),
                )
                if (response.success) {
                    if (qrBytes != null && !uploadPayoutQr(token, id, qrBytes)) {
                        // The patch DID land — refresh so the list reflects it.
                        _actionState.value =
                            ActionState.Failed("Payout was updated, but its QR could not be uploaded.")
                        fetchSettings()
                        return@launch
                    }
                    _actionState.value = ActionState.Done("Payout updated.")
                    fetchSettings()
                } else {
                    _actionState.value = ActionState.Failed(response.error ?: "Failed to update payout.")
                }
            } catch (e: Exception) {
                _actionState.value = ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun deletePayoutAccount(id: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _actionState.value = ActionState.Busy("Removing payout...")
            try {
                val response = RetrofitClient.apiService.deletePayout("Bearer $token", id)
                if (response.success) {
                    _actionState.value = ActionState.Done("Payout removed.")
                    fetchSettings()
                } else {
                    _actionState.value = ActionState.Failed(response.error ?: "Failed to remove payout.")
                }
            } catch (e: Exception) {
                _actionState.value = ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    fun setPrimaryPayoutAccount(id: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            _actionState.value = ActionState.Busy("Updating primary…")
            try {
                val response = RetrofitClient.apiService.setPrimaryPayout("Bearer $token", id)
                if (response.success && response.data != null) {
                    _payoutAccounts.value = response.data
                    _actionState.value = ActionState.Done("Primary updated.")
                } else {
                    _actionState.value = ActionState.Failed(response.error ?: "Failed to set primary.")
                }
            } catch (e: Exception) {
                _actionState.value = ActionState.Failed(e.localizedMessage ?: "Connection error.")
            }
        }
    }

    private suspend fun uploadPayoutQr(token: String, id: String, bytes: ByteArray): Boolean {
        val request = bytes.toRequestBody("image/png".toMediaTypeOrNull(), 0, bytes.size)
        val part = MultipartBody.Part.createFormData("file", "qr.png", request)
        // A thrown HTTP error (e.g. 429 — the QR route shares the backend's
        // 20/min media-upload bucket) must land on the callers' "payout was
        // added, but…" branch and its fetchSettings(), not skip into the
        // generic catch. Both failure shapes → false; cancellation still
        // propagates.
        return try {
            RetrofitClient.apiService.uploadPayoutQr("Bearer $token", id, part).success
        } catch (e: CancellationException) {
            throw e
        } catch (_: Exception) {
            false
        }
    }
}
