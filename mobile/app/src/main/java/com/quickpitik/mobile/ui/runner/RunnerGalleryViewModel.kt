package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.QpWebSocket
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.SelfieRefDto
import com.quickpitik.mobile.data.remote.SearchByFaceJsonRequest
import com.quickpitik.mobile.data.remote.PhotoAlertRequest
import com.quickpitik.mobile.data.remote.WsFrameEnvelope
import com.quickpitik.mobile.data.remote.WsState
import com.quickpitik.mobile.data.local.SessionManager
import android.net.Uri
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File

sealed class RunnerEventsState {
    object Loading : RunnerEventsState()
    data class Success(val events: List<EventDto>) : RunnerEventsState()
    data class Error(val message: String) : RunnerEventsState()
}

sealed class PhotosSearchState {
    object Idle : PhotosSearchState()
    object Loading : PhotosSearchState()
    data class Success(val photos: List<PhotoDto>) : PhotosSearchState()
    data class Error(val message: String) : PhotosSearchState()
}

// "Notify me when my photos are ready" opt-in state for the active event.
// Mobile is always authed here, so the only fork is has-selfie (togglable) vs
// no-selfie (prompt to add one).
sealed class PhotoAlertUiState {
    object Loading : PhotoAlertUiState()
    object NeedsSelfie : PhotoAlertUiState()
    data class Ready(val registered: Boolean, val updating: Boolean = false) : PhotoAlertUiState()
}

class RunnerGalleryViewModel(application: Application) : AndroidViewModel(application) {
    private val _eventsState = MutableStateFlow<RunnerEventsState>(RunnerEventsState.Loading)
    val eventsState: StateFlow<RunnerEventsState> = _eventsState

    private val _activeEvent = MutableStateFlow<EventDto?>(null)
    val activeEvent: StateFlow<EventDto?> = _activeEvent

    private val _searchState = MutableStateFlow<PhotosSearchState>(PhotosSearchState.Idle)
    val searchState: StateFlow<PhotosSearchState> = _searchState

    private val _isFiltered = MutableStateFlow(false)
    val isFiltered: StateFlow<Boolean> = _isFiltered

    private val _photoAlert = MutableStateFlow<PhotoAlertUiState>(PhotoAlertUiState.Loading)
    val photoAlert: StateFlow<PhotoAlertUiState> = _photoAlert

    // ---- Live photo arrival (/ws/events/{id}/photos) ------------------------
    //
    // The website prepends each pushed photo straight into its query cache. We
    // deliberately do NOT: the push frame carries only
    // {id, bib, tone, span, imageUrl, uploadedAt}, while PhotoDto types `time`
    // and `price` non-null. A prepended row would render "₱0.00" in the
    // lightbox (PhotoPreview) and could be added to the cart at that price.
    // Instead the frame is a signal — count it, then re-run the search that's
    // already on screen so every field comes from the authoritative REST DTO.
    // Net effect for the runner is identical: photos land on their own.

    private val liveSocket = QpWebSocket(SessionManager.getInstance(application), viewModelScope)
    private val gson = Gson()

    private val _newPhotoCount = MutableStateFlow(0)
    val newPhotoCount: StateFlow<Int> = _newPhotoCount

    val liveState: StateFlow<WsState> = liveSocket.state

    // The bib query currently on screen. searchByBib() takes it as a parameter
    // and only a boolean survived in _isFiltered, so an auto-refresh had no way
    // to re-run the SAME search without this.
    private var activeBibQuery: String = ""

    // Face-search results live in the same _searchState as bib results, so an
    // unguarded auto-refresh would silently replace a runner's match set with
    // the unfiltered event list. In face mode we only count.
    private var isFaceSearchMode: Boolean = false

    private var liveRefreshJob: Job? = null

    // Whether this channel has been open before for the CURRENT event. The very
    // first open follows selectEvent's own fetch, so refetching then would just
    // double a request; only a RE-open (backgrounded, dropped Wi-Fi) has a gap
    // to close. Reset whenever the selected event changes.
    private var liveChannelHasOpened = false

    init {
        fetchPublicEvents()
        observeLivePhotos()
    }

    private fun observeLivePhotos() {
        viewModelScope.launch {
            liveSocket.state.collect { state ->
                if (state !is WsState.Open) return@collect
                // Reconnect grace: anything published while the socket was down
                // was never pushed to us. Mirrors the inbox channels and the
                // website's ws.onopen refetch.
                if (liveChannelHasOpened && !isFaceSearchMode) {
                    runBibSearch(activeBibQuery, showLoading = false)
                }
                liveChannelHasOpened = true
            }
        }
        viewModelScope.launch {
            liveSocket.frames.collect { raw ->
                val type = runCatching {
                    gson.fromJson(raw, WsFrameEnvelope::class.java)?.type
                }.getOrNull()
                // photo.indexed rides the same channel but only carries
                // {id, bib, indexingStatus} and mobile has no per-photo indexing
                // UI — drop it, and anything else we don't know, quietly.
                if (type != "photo.published") return@collect
                _newPhotoCount.value = _newPhotoCount.value + 1
                if (isFaceSearchMode) return@collect
                // Debounced: a photographer uploading a burst would otherwise
                // fire one refetch per photo mid-race.
                liveRefreshJob?.cancel()
                liveRefreshJob = viewModelScope.launch {
                    delay(LIVE_REFRESH_DEBOUNCE_MS)
                    runBibSearch(activeBibQuery, showLoading = false)
                }
            }
        }
    }

    /**
     * Opens the push channel for the selected event. Driven by the cockpit's
     * lifecycle rather than by [selectEvent], so a backgrounded app isn't
     * holding a socket open in the runner's pocket for a 3-hour race.
     *
     * No-op unless the event is LIVE — only live events push. Same gate the
     * website uses (`state === "live"`), via the helper [selectEvent] already
     * branches on.
     */
    fun connectLivePhotos() {
        val event = _activeEvent.value ?: return
        if (deriveEventState(event.date) != EventState.LIVE) return
        liveSocket.connect("/ws/events/${event.id}/photos")
    }

    fun disconnectLivePhotos() {
        liveSocket.close()
        liveRefreshJob?.cancel()
    }

    /** Pill tap — the runner has seen them. */
    fun resetNewPhotoCount() {
        _newPhotoCount.value = 0
    }

    /**
     * "Connection lost · Refresh" affordance: pull the current list by hand and
     * start the socket over from a clean attempt count.
     */
    fun retryLivePhotos() {
        _newPhotoCount.value = 0
        if (_activeEvent.value == null) return
        if (!isFaceSearchMode) runBibSearch(activeBibQuery, showLoading = false)
        // Force the socket shut rather than dropping one subscriber: this resets
        // the attempt counter, so the retry starts from a clean backoff instead
        // of waiting out the 30s cap it had climbed to. connectLivePhotos()
        // below restores the cockpit's own subscription.
        liveRefreshJob?.cancel()
        liveSocket.release()
        connectLivePhotos()
    }

    override fun onCleared() {
        liveSocket.release()
        super.onCleared()
    }

    fun fetchPublicEvents() {
        viewModelScope.launch {
            // Only show the skeleton on first load; re-fetches (screen re-entry,
            // future pull-to-refresh) keep the existing list visible so the UI
            // doesn't flash back to a loading state on every navigation.
            if (_eventsState.value !is RunnerEventsState.Success) {
                _eventsState.value = RunnerEventsState.Loading
            }
            try {
                // Browse-first, like the website's /events: pull every publicly
                // visible lifecycle bucket so the discovery screen can segment
                // Upcoming / Live / Recent / Archive. Selection is explicit (a
                // card tap calls selectEvent) — we no longer auto-select the first
                // event, since the runner now lands on the browse screen, not the
                // cockpit. Backend EventController.parseStatusList accepts the CSV.
                val response = RetrofitClient.apiService.getPublicEvents(
                    status = "ACTIVE,COMPLETED,ARCHIVED",
                    offset = 0,
                    limit = 200
                )
                if (response.success && response.data != null) {
                    _eventsState.value = RunnerEventsState.Success(response.data.items)
                } else {
                    _eventsState.value = RunnerEventsState.Error(response.error ?: "Failed to load events.")
                }
            } catch (e: Exception) {
                _eventsState.value = RunnerEventsState.Error(e.localizedMessage ?: "Failed to connect to backend server.")
            }
        }
    }

    // Resolve a loaded event by slug. Used by the profile race log's "Open →" to
    // jump straight into the cockpit for a saved/purchased event without a second
    // round-trip. Returns null if events haven't loaded yet or there's no match.
    fun eventBySlug(slug: String): EventDto? =
        (_eventsState.value as? RunnerEventsState.Success)?.events?.firstOrNull { it.slug == slug }

    fun selectEvent(event: EventDto) {
        _activeEvent.value = event
        _isFiltered.value = false
        // Only channel STATE is reset here; the socket itself is owned by the
        // cockpit's lifecycle effect, which is keyed on the active event id and
        // so tears down / reopens on its own when this value changes. A pending
        // refresh must not fire against the newly selected event.
        liveRefreshJob?.cancel()
        liveChannelHasOpened = false
        _newPhotoCount.value = 0
        // Pre-race-day events have no gallery yet — the website never loads
        // photos for one, and the cockpit renders an "opens on race day" notice
        // instead. Skipping the fetch keeps the state Idle so that branch shows
        // cleanly rather than flashing an empty result.
        if (deriveEventState(event.date) == EventState.UPCOMING) {
            _searchState.value = PhotosSearchState.Idle
            return
        }
        searchByBib("")
    }

    fun clearSelectedEvent() {
        _activeEvent.value = null
        _searchState.value = PhotosSearchState.Idle
        _isFiltered.value = false
        // As in selectEvent: the lifecycle effect closes the socket when the
        // active event id changes. Here we only drop the derived state.
        liveRefreshJob?.cancel()
        liveChannelHasOpened = false
        _newPhotoCount.value = 0
    }

    fun searchByBib(bib: String) {
        runBibSearch(bib, showLoading = true)
    }

    /**
     * [showLoading] is false for the live-photo auto-refresh: the runner didn't
     * ask for anything, so swapping a populated grid for skeletons every time a
     * photo lands would read as the app breaking, not updating.
     */
    private fun runBibSearch(bib: String, showLoading: Boolean) {
        val event = _activeEvent.value ?: return
        val trimmed = bib.trim()
        activeBibQuery = trimmed
        isFaceSearchMode = false
        _isFiltered.value = trimmed.isNotEmpty()
        viewModelScope.launch {
            if (showLoading) _searchState.value = PhotosSearchState.Loading
            try {
                val query = trimmed.ifEmpty { null }
                val response = RetrofitClient.apiService.getEventPhotos(
                    slug = event.slug,
                    bib = query
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(response.data.items)
                } else if (showLoading) {
                    _searchState.value = PhotosSearchState.Error(response.error ?: "Search lookup failed.")
                }
            } catch (e: Exception) {
                // A failed background refresh keeps the photos already on screen
                // — only a user-initiated search is allowed to surface an error.
                if (showLoading) {
                    _searchState.value = PhotosSearchState.Error(RetrofitClient.parseError(e))
                }
            }
        }
    }

    fun clearFilter() {
        _isFiltered.value = false
        searchByBib("")
    }

    fun searchBySelfie(selfieFile: File) {
        val event = _activeEvent.value ?: return
        _isFiltered.value = true
        // Face results share _searchState with bib results — mark the mode so a
        // live-photo push can't refresh this match set out from under the runner.
        isFaceSearchMode = true
        val application = getApplication<Application>()
        val sessionManager = SessionManager.getInstance(application)
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _searchState.value = PhotosSearchState.Error("Authentication token not found. Please log in.")
            return
        }
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                val requestFile = selfieFile.asRequestBody("image/jpeg".toMediaTypeOrNull())
                val selfiePart = MultipartBody.Part.createFormData("selfie", selfieFile.name, requestFile)

                val response = RetrofitClient.apiService.searchPhotosByFace(
                    token = "Bearer $token",
                    slug = event.slug,
                    selfie = selfiePart
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(response.data.items)
                } else {
                    _searchState.value = PhotosSearchState.Error(response.error ?: "AI Face Recognition returned error.")
                }
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    // Real selfie capture/upload path: read the bytes the camera (or gallery
    // picker) handed us via the content URI, spool them to a cache file, then
    // run the same multipart face-search the stored-selfie path uses.
    fun searchBySelfieUri(uri: Uri) {
        _activeEvent.value ?: return
        isFaceSearchMode = true
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                val resolver = getApplication<Application>().contentResolver
                val bytes = resolver.openInputStream(uri)?.use { it.readBytes() }
                if (bytes == null || bytes.isEmpty()) {
                    _searchState.value = PhotosSearchState.Error("Couldn't read that photo. Please try again.")
                    return@launch
                }
                val cacheFile = File(
                    getApplication<Application>().cacheDir,
                    "selfie_search_${System.currentTimeMillis()}.jpg"
                )
                cacheFile.writeBytes(bytes)
                searchBySelfie(cacheFile)
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(e.localizedMessage ?: "Failed to process the selfie image.")
            }
        }
    }

    fun searchByStoredSelfie() {
        val event = _activeEvent.value ?: return
        _isFiltered.value = true
        isFaceSearchMode = true
        val application = getApplication<Application>()
        val sessionManager = SessionManager.getInstance(application)
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _searchState.value = PhotosSearchState.Error("Authentication token not found. Please log in.")
            return
        }
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                // 1. Fetch user's selfies to find the primary selfie ID
                val selfiesResponse = RetrofitClient.apiService.getSelfies("Bearer $token")
                if (!selfiesResponse.success || selfiesResponse.data == null) {
                    _searchState.value = PhotosSearchState.Error(selfiesResponse.error ?: "Failed to retrieve user selfies.")
                    return@launch
                }
                
                val primarySelfie = selfiesResponse.data.find { it.isPrimary }
                if (primarySelfie == null) {
                    _searchState.value = PhotosSearchState.Error("No primary selfie set. Please upload a selfie and set it as primary first.")
                    return@launch
                }
                
                // 2. Perform the JSON search using the primary selfie's ID
                val response = RetrofitClient.apiService.searchPhotosByFaceJson(
                    token = "Bearer $token",
                    slug = event.slug,
                    request = SearchByFaceJsonRequest(selfieId = primarySelfie.id)
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(response.data.items)
                } else {
                    _searchState.value = PhotosSearchState.Error(response.error ?: "AI Face search returned an error.")
                }
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(e.localizedMessage ?: "Failed to query event photos with stored selfie.")
            }
        }
    }

    // Loads the opt-in state for the active event. Reuses the selfie library the
    // face search does: no selfie → NeedsSelfie (prompt to add one); otherwise
    // Ready(registered) from GET /photo-alert.
    fun loadPhotoAlert(slug: String) {
        viewModelScope.launch {
            _photoAlert.value = PhotoAlertUiState.Loading
            val token = SessionManager.getInstance(getApplication()).getAccessToken()
            if (token == null) {
                _photoAlert.value = PhotoAlertUiState.NeedsSelfie
                return@launch
            }
            try {
                val selfies = RetrofitClient.apiService.getSelfies("Bearer $token")
                val hasSelfie = selfies.success && !selfies.data.isNullOrEmpty()
                if (!hasSelfie) {
                    _photoAlert.value = PhotoAlertUiState.NeedsSelfie
                    return@launch
                }
                val status = RetrofitClient.apiService.getPhotoAlertStatus("Bearer $token", slug)
                val registered = status.success && status.data?.registered == true
                _photoAlert.value = PhotoAlertUiState.Ready(registered = registered)
            } catch (e: Exception) {
                // Couldn't confirm — still let the runner try; a genuinely
                // selfie-less register surfaces as NeedsSelfie in togglePhotoAlert.
                _photoAlert.value = PhotoAlertUiState.Ready(registered = false)
            }
        }
    }

    fun togglePhotoAlert(slug: String, register: Boolean) {
        val current = _photoAlert.value as? PhotoAlertUiState.Ready ?: return
        viewModelScope.launch {
            _photoAlert.value = current.copy(updating = true)
            val token = SessionManager.getInstance(getApplication()).getAccessToken()
            if (token == null) {
                _photoAlert.value = current
                return@launch
            }
            try {
                if (register) {
                    val resp = RetrofitClient.apiService.registerPhotoAlert(
                        "Bearer $token", slug, PhotoAlertRequest()
                    )
                    _photoAlert.value =
                        if (resp.success) PhotoAlertUiState.Ready(registered = true) else current
                } else {
                    RetrofitClient.apiService.unregisterPhotoAlert("Bearer $token", slug)
                    _photoAlert.value = PhotoAlertUiState.Ready(registered = false)
                }
            } catch (e: Exception) {
                // A 400 SELFIE_REQUIRED means the selfie vanished between load and
                // toggle — route them to add one; anything else just reverts.
                val msg = RetrofitClient.parseError(e)
                _photoAlert.value =
                    if (msg.contains("selfie", ignoreCase = true)) PhotoAlertUiState.NeedsSelfie
                    else current
            }
        }
    }

    private companion object {
        // Long enough to collapse a burst upload into one refetch, short enough
        // that a single shot still feels immediate.
        const val LIVE_REFRESH_DEBOUNCE_MS = 1_500L
    }
}
