package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.isPhotographerRole
import com.quickpitik.mobile.data.remote.PhotographerEventDetailDto
import com.quickpitik.mobile.data.remote.PhotographerLibraryPhotoDto
import com.quickpitik.mobile.data.remote.PhotographerPurchaseDto
import com.quickpitik.mobile.data.remote.PhotographerPurchasedPhotoDto
import com.quickpitik.mobile.data.remote.PurchaseEventDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.launch
import retrofit2.HttpException

data class PurchaseEventState(
    val eventId: String? = null,
    val event: PhotographerEventDetailDto? = null,
    val isLoading: Boolean = false,
    val error: String? = null,
    val errorStatus: Int? = null,
)

data class PurchasedPhotosState(
    val eventId: String? = null,
    val items: List<PhotographerPurchasedPhotoDto> = emptyList(),
    val total: Long = 0,
    val nextOffset: Int = 0,
    val isLoading: Boolean = false,
    val isLoadingMore: Boolean = false,
    val error: String? = null,
    val appendError: String? = null,
    val errorStatus: Int? = null,
    val hasMore: Boolean = true,
)

data class PurchasePreviewState(
    val photoId: String? = null,
    val orderId: String? = null,
    val transactionId: String? = null,
    val photo: PhotographerLibraryPhotoDto? = null,
    val event: PurchaseEventDto? = null,
    val items: List<PhotographerPurchaseDto> = emptyList(),
    val selectedPurchase: PhotographerPurchaseDto? = null,
    val total: Long = 0,
    val nextOffset: Int = 0,
    val isLoading: Boolean = false,
    val isLoadingMore: Boolean = false,
    val error: String? = null,
    val appendError: String? = null,
    val errorStatus: Int? = null,
    val hasMore: Boolean = true,
)

/** Read-only purchase history; loads only routes the photographer opens. */
class PhotographerPurchasesViewModel(application: Application) : AndroidViewModel(application) {
    private val session = SessionManager.getInstance(application)
    private val _eventState = MutableStateFlow(PurchaseEventState())
    val eventState: StateFlow<PurchaseEventState> = _eventState
    private val _purchasedPhotosState = MutableStateFlow(PurchasedPhotosState())
    val purchasedPhotosState: StateFlow<PurchasedPhotosState> = _purchasedPhotosState
    private val _previewState = MutableStateFlow(PurchasePreviewState())
    val previewState: StateFlow<PurchasePreviewState> = _previewState

    private var eventJob: Job? = null
    private var galleryJob: Job? = null
    private var previewJob: Job? = null
    private var eventRequest = 0L
    private var galleryRequest = 0L
    private var previewRequest = 0L

    init {
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            session.authChanges.drop(1).collect { clearPrivateState() }
        }
    }

    private fun bearer(): String? = session.getAccessToken()
        ?.takeIf { isPhotographerRole(session.getUserRole()) }
        ?.let { "Bearer $it" }

    private fun current(generation: Long) = generation == session.authChanges.value && bearer() != null
    private fun status(e: Exception): Int? = (e as? HttpException)?.code()
    private fun message(e: Exception) = RetrofitClient.parseHttpError(e)?.message ?: RetrofitClient.parseError(e)

    private fun clearPrivateState() {
        eventRequest++
        galleryRequest++
        previewRequest++
        eventJob?.cancel()
        galleryJob?.cancel()
        previewJob?.cancel()
        _eventState.value = PurchaseEventState()
        _purchasedPhotosState.value = PurchasedPhotosState(hasMore = false)
        _previewState.value = PurchasePreviewState(hasMore = false)
    }

    fun loadEvent(eventId: String) {
        if (eventId != _eventState.value.eventId) {
            galleryRequest++
            galleryJob?.cancel()
            _purchasedPhotosState.value = PurchasedPhotosState(hasMore = false)
            closePreview()
        }
        eventRequest++
        eventJob?.cancel()
        val request = eventRequest
        val generation = session.authChanges.value
        val token = bearer() ?: return
        _eventState.value = PurchaseEventState(eventId = eventId, isLoading = true)
        eventJob = viewModelScope.launch {
            try {
                val response = RetrofitClient.apiService.getPhotographerEventDetail(token, eventId, includeRemoved = true)
                if (request != eventRequest || !current(generation)) return@launch
                _eventState.value = if (response.success && response.data != null)
                    PurchaseEventState(eventId, response.data)
                else PurchaseEventState(eventId, error = response.error ?: "Couldn't load event.")
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (request == eventRequest && current(generation))
                    _eventState.value = PurchaseEventState(eventId, error = message(e), errorStatus = status(e))
            }
        }
    }

    fun loadPurchasedPhotos(eventId: String, refresh: Boolean = false) {
        val previous = _purchasedPhotosState.value
        if (!refresh && previous.eventId == eventId && (previous.isLoading || previous.items.isNotEmpty())) return
        galleryRequest++
        galleryJob?.cancel()
        val request = galleryRequest
        val generation = session.authChanges.value
        val token = bearer() ?: return
        _purchasedPhotosState.value = PurchasedPhotosState(eventId, isLoading = true)
        galleryJob = viewModelScope.launch {
            try {
                val response = RetrofitClient.apiService.getPhotographerPurchasedPhotos(token, eventId)
                if (request != galleryRequest || !current(generation)) return@launch
                val page = response.data.takeIf { response.success } ?: error(response.error ?: "Couldn't load purchased photos.")
                val next = page.offset + page.limit
                _purchasedPhotosState.value = PurchasedPhotosState(eventId, page.items.distinctBy { it.photo.id }, page.total, next, hasMore = page.items.isNotEmpty() && next < page.total)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (request == galleryRequest && current(generation))
                    _purchasedPhotosState.value = PurchasedPhotosState(eventId, error = message(e), errorStatus = status(e))
            }
        }
    }

    fun loadMorePurchasedPhotos() {
        val previous = _purchasedPhotosState.value
        if (previous.eventId == null || previous.isLoading || previous.isLoadingMore || !previous.hasMore) return
        val request = galleryRequest
        val generation = session.authChanges.value
        val token = bearer() ?: return
        _purchasedPhotosState.value = previous.copy(isLoadingMore = true, appendError = null, errorStatus = null)
        galleryJob = viewModelScope.launch {
            try {
                val response = RetrofitClient.apiService.getPhotographerPurchasedPhotos(token, previous.eventId, previous.nextOffset)
                if (request != galleryRequest || !current(generation)) return@launch
                val page = response.data.takeIf { response.success } ?: error(response.error ?: "Couldn't load more purchased photos.")
                val next = page.offset + page.limit
                val latest = _purchasedPhotosState.value
                _purchasedPhotosState.value = latest.copy(items = (latest.items + page.items).distinctBy { it.photo.id }, total = page.total, nextOffset = next, isLoadingMore = false, hasMore = page.items.isNotEmpty() && next < page.total)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (request == galleryRequest && current(generation))
                    _purchasedPhotosState.value = _purchasedPhotosState.value.copy(isLoadingMore = false, appendError = message(e), errorStatus = status(e))
            }
        }
    }

    fun openPurchase(photoId: String, orderId: String? = null, transactionId: String? = null) {
        previewRequest++
        previewJob?.cancel()
        val request = previewRequest
        val generation = session.authChanges.value
        val token = bearer() ?: return
        _previewState.value = PurchasePreviewState(photoId, orderId, transactionId, isLoading = true)
        previewJob = viewModelScope.launch {
            try {
                val response = RetrofitClient.apiService.getPhotographerPhotoPurchases(token, photoId, orderId)
                if (request != previewRequest || !current(generation)) return@launch
                val page = response.data.takeIf { response.success } ?: error(response.error ?: "Couldn't load purchases.")
                val next = page.offset + page.limit
                val selected = if (orderId == null) page.selectedPurchase ?: page.items.firstOrNull()
                    else page.selectedPurchase?.takeIf { it.orderId == orderId } ?: page.items.find { it.orderId == orderId }
                _previewState.value = PurchasePreviewState(photoId, orderId, transactionId, page.photo, page.event,
                    page.items.distinctBy { it.orderId to it.photoId }, selected,
                    page.total, next, hasMore = page.items.isNotEmpty() && next < page.total)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (request == previewRequest && current(generation))
                    _previewState.value = PurchasePreviewState(photoId, orderId, transactionId, error = message(e), errorStatus = status(e))
            }
        }
    }

    fun loadMorePurchases() {
        val previous = _previewState.value
        if (previous.photoId == null || previous.isLoading || previous.isLoadingMore || !previous.hasMore) return
        val request = previewRequest
        val generation = session.authChanges.value
        val token = bearer() ?: return
        _previewState.value = previous.copy(isLoadingMore = true, appendError = null, errorStatus = null)
        previewJob = viewModelScope.launch {
            try {
                val response = RetrofitClient.apiService.getPhotographerPhotoPurchases(token, previous.photoId, previous.orderId, previous.nextOffset)
                if (request != previewRequest || !current(generation)) return@launch
                val page = response.data.takeIf { response.success } ?: error(response.error ?: "Couldn't load more purchases.")
                val next = page.offset + page.limit
                val latest = _previewState.value
                _previewState.value = latest.copy(items = (latest.items + page.items).distinctBy { it.orderId to it.photoId }, total = page.total, nextOffset = next, isLoadingMore = false, hasMore = page.items.isNotEmpty() && next < page.total)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (request == previewRequest && current(generation))
                    _previewState.value = _previewState.value.copy(isLoadingMore = false, appendError = message(e), errorStatus = status(e))
            }
        }
    }

    fun selectPurchase(orderId: String) {
        val state = _previewState.value
        val local = state.items.find { it.orderId == orderId }
        if (local != null) _previewState.value = state.copy(orderId = orderId, selectedPurchase = local)
        else state.photoId?.let { openPurchase(it, orderId, state.transactionId) }
    }

    fun retryPreview() {
        val state = _previewState.value
        state.photoId?.let { openPurchase(it, state.orderId, state.transactionId) }
    }

    fun closePreview() {
        previewRequest++
        previewJob?.cancel()
        _previewState.value = PurchasePreviewState(hasMore = false)
    }
}
