package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.quickpitik.mobile.data.remote.EventDetailDto
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
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
import com.quickpitik.mobile.data.readAtMost
import android.net.Uri
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
    data class Success(
        val photos: List<PhotoDto>,
        // Server-side total for "Showing first N of M" + the Load more CTA.
        // Face-search results are one-shot (total == photos.size).
        val total: Long = 0,
        val loadingMore: Boolean = false,
    ) : PhotosSearchState()
    data class Error(val message: String) : PhotosSearchState()
}

// "Notify me when my photos are ready" opt-in state for the active event.
// Mobile is always authed here, so the only fork is has-selfie (togglable) vs
// no-selfie (prompt to add one).
sealed class PhotoAlertUiState {
    object Loading : PhotoAlertUiState()
    object NeedsSelfie : PhotoAlertUiState()
    data class Ready(
        val registered: Boolean,
        val updating: Boolean = false,
        // Set when a toggle attempt failed for a reason other than a missing
        // selfie (e.g. EVENT_NOT_UPLOADABLE after the alert window closed) —
        // the card must say why the switch snapped back, not revert silently.
        val message: String? = null,
    ) : PhotoAlertUiState()
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

    // Editorial detail for the AboutStrip (organizer, description, categories,
    // pricing) — GET /events/{slug}, fetched best-effort on selectEvent. Null
    // until loaded (or on failure): the strip simply doesn't render, the
    // cockpit works without it.
    private val _eventDetail = MutableStateFlow<EventDetailDto?>(null)
    val eventDetail: StateFlow<EventDetailDto?> = _eventDetail

    // Selfie library for the cockpit's face-search picker — the runner picks
    // WHICH selfie to match with (web SelfieSearchPanel parity), instead of
    // the old silent primary-only button.
    private val _selfies = MutableStateFlow<List<SelfieRefDto>>(emptyList())
    val selfies: StateFlow<List<SelfieRefDto>> = _selfies

    fun loadGalleryMetadata(photoAlertSlug: String?) {
        viewModelScope.launch {
            if (photoAlertSlug != null) _photoAlert.value = PhotoAlertUiState.Loading
            val token = SessionManager.getInstance(getApplication()).getAccessToken()
            if (token == null) {
                if (photoAlertSlug != null) _photoAlert.value = PhotoAlertUiState.NeedsSelfie
                return@launch
            }
            try {
                // Only overwrite the cache on a real payload — a transient
                // success=false envelope must not blank the picker grid or flip
                // the alert card to NeedsSelfie for a registered runner.
                val response = RetrofitClient.apiService.getSelfies("Bearer $token")
                if (response.success && response.data != null) _selfies.value = response.data
                val selfies = _selfies.value
                if (photoAlertSlug == null) return@launch
                if (selfies.isEmpty()) {
                    _photoAlert.value = PhotoAlertUiState.NeedsSelfie
                    return@launch
                }
                val status = RetrofitClient.apiService.getPhotoAlertStatus("Bearer $token", photoAlertSlug)
                _photoAlert.value = PhotoAlertUiState.Ready(
                    registered = status.success && status.data?.registered == true,
                )
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                if (photoAlertSlug != null) {
                    _photoAlert.value = PhotoAlertUiState.Ready(registered = false)
                }
            }
        }
    }

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

    // Returns the Job so pull-to-refresh can join() it and settle its spinner
    // when the fetch actually finishes (existing fire-and-forget callers are
    // unaffected).
    fun fetchPublicEvents(): Job = viewModelScope.launch {
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

    // Resolve a loaded event by slug. Used by the profile race log's "Open →" to
    // jump straight into the cockpit for a saved/purchased event without a second
    // round-trip. Returns null if events haven't loaded yet or there's no match.
    fun eventBySlug(slug: String): EventDto? =
        (_eventsState.value as? RunnerEventsState.Success)?.events?.firstOrNull { it.slug == slug }

    fun selectEvent(event: EventDto) {
        _activeEvent.value = event
        _isFiltered.value = false
        // Editorial detail is best-effort and non-blocking — the cockpit
        // renders from the list DTO while this fills the AboutStrip in.
        _eventDetail.value = null
        viewModelScope.launch {
            runCatching { RetrofitClient.apiService.getEventDetail(event.slug) }
                .getOrNull()
                ?.takeIf { it.success && it.data != null }
                ?.let { detail ->
                    // Guard against a stale response after a rapid re-select.
                    if (_activeEvent.value?.slug == event.slug) _eventDetail.value = detail.data
                }
        }
        // Only channel STATE is reset here; the socket itself is owned by the
        // cockpit's lifecycle effect, which is keyed on the active event id and
        // so tears down / reopens on its own when this value changes. A pending
        // refresh must not fire against the newly selected event.
        liveRefreshJob?.cancel()
        bibSearchJob?.cancel()
        bibRequestJob?.cancel()
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
        // Direct, not via searchByBib(): the debounce there exists for
        // keystrokes; the initial load should not wait 350ms.
        runBibSearch("", showLoading = true)
    }

    fun clearSelectedEvent() {
        _activeEvent.value = null
        _searchState.value = PhotosSearchState.Idle
        _isFiltered.value = false
        _eventDetail.value = null
        // As in selectEvent: the lifecycle effect closes the socket when the
        // active event id changes. Here we only drop the derived state.
        liveRefreshJob?.cancel()
        bibSearchJob?.cancel()
        bibRequestJob?.cancel()
        liveChannelHasOpened = false
        _newPhotoCount.value = 0
    }

    // Debounced: the cockpit's bib field calls this on every keystroke, and
    // "1234" must not fire four network searches. Mirrors the live-photo
    // refresh debounce below; a submitted query still lands within ~a third of
    // a second of the last key.
    private var bibSearchJob: Job? = null
    private var bibRequestJob: Job? = null

    fun searchByBib(bib: String) {
        bibSearchJob?.cancel()
        bibRequestJob?.cancel()
        bibSearchJob = viewModelScope.launch {
            delay(BIB_SEARCH_DEBOUNCE_MS)
            runBibSearch(bib, showLoading = true)
        }
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
        bibRequestJob?.cancel()
        bibRequestJob = viewModelScope.launch {
            if (showLoading) _searchState.value = PhotosSearchState.Loading
            try {
                val query = trimmed.ifEmpty { null }
                // Bearer when signed in (always, on mobile): the backend uses it
                // to populate cleanUrl for owned photos and to rate-bucket per
                // user — see the QuickPitikApi declaration.
                val token = SessionManager.getInstance(getApplication()).getAccessToken()
                val response = RetrofitClient.apiService.getEventPhotos(
                    token = token?.let { "Bearer $it" },
                    slug = event.slug,
                    bib = query
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(
                        photos = response.data.items,
                        total = response.data.total,
                    )
                } else if (showLoading) {
                    _searchState.value = PhotosSearchState.Error(response.error ?: "Search lookup failed.")
                }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
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
        // Explicit tap — no reason to sit out the keystroke debounce.
        bibSearchJob?.cancel()
        bibRequestJob?.cancel()
        runBibSearch("", showLoading = true)
    }

    /**
     * Appends the next page of the CURRENT bib/browse query ("Load more").
     * Face-search results are one-shot and never paged, matching the website.
     */
    fun loadMorePhotos() {
        val event = _activeEvent.value ?: return
        val current = _searchState.value as? PhotosSearchState.Success ?: return
        if (isFaceSearchMode || current.loadingMore) return
        if (current.photos.size >= current.total) return
        viewModelScope.launch {
            _searchState.value = current.copy(loadingMore = true)
            try {
                val token = SessionManager.getInstance(getApplication()).getAccessToken()
                val response = RetrofitClient.apiService.getEventPhotos(
                    token = token?.let { "Bearer $it" },
                    slug = event.slug,
                    bib = activeBibQuery.ifEmpty { null },
                    offset = current.photos.size,
                )
                if (response.success && response.data != null) {
                    // Dedupe by id: a photo published between the two pages
                    // shifts offsets and could deliver a duplicate row.
                    val seen = current.photos.mapTo(HashSet()) { it.id }
                    _searchState.value = PhotosSearchState.Success(
                        photos = current.photos + response.data.items.filter { it.id !in seen },
                        total = response.data.total,
                    )
                } else {
                    _searchState.value = current.copy(loadingMore = false)
                }
            } catch (e: Exception) {
                // Keep what's on screen; the CTA simply becomes tappable again.
                _searchState.value = current.copy(loadingMore = false)
            }
        }
    }

    // Website-parity copy for face-search failures (bib-search-panels.tsx
    // humanizeError). The backend passes ai-api codes through the envelope
    // specifically so clients can map them to targeted copy; the raw server
    // message is the fallback, never the raw exception.
    private fun humanizeFaceSearchError(code: String?, message: String?): String = when (code) {
        "LOW_CONFIDENCE" -> "We didn't find your face in this event yet. Try another shot."
        "SELFIE_REJECTED", "LOW_QUALITY", "NO_FACES" ->
            message ?: "Selfie rejected — try a clearer, frontal shot."
        "AI_API_UNAVAILABLE" -> "Face search is offline right now. Try again in a few minutes."
        else -> message ?: "Could not match your face right now. Try again."
    }

    // For thrown (non-2xx) failures. parseHttpError drains the error body, so
    // parseError is only consulted when there was no structured error to read
    // (transport failures — where its human copy is exactly what we want).
    private fun faceSearchErrorFrom(e: Exception): String {
        val err = RetrofitClient.parseHttpError(e)
        return if (err != null) humanizeFaceSearchError(err.code, err.message)
        else RetrofitClient.parseError(e)
    }

    private suspend fun searchBySelfie(selfieFile: File) {
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
                    _searchState.value = PhotosSearchState.Success(
                        photos = response.data.items,
                        total = response.data.items.size.toLong(),
                    )
                } else {
                    val err = response.errors?.firstOrNull()
                    _searchState.value =
                        PhotosSearchState.Error(humanizeFaceSearchError(err?.code, err?.message))
                }
        } catch (e: Exception) {
            if (e is CancellationException) throw e
            _searchState.value = PhotosSearchState.Error(faceSearchErrorFrom(e))
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
            var cacheFile: File? = null
            try {
                val resolver = getApplication<Application>().contentResolver
                val file = withContext(Dispatchers.IO) {
                    val target = File.createTempFile(
                        "selfie_search_",
                        ".jpg",
                        getApplication<Application>().cacheDir,
                    )
                    val copied = resolver.openInputStream(uri)?.use { input ->
                        val bytes = input.readAtMost(MAX_UPLOAD_BYTES + 1)
                        if (bytes.size > MAX_UPLOAD_BYTES) 0L else {
                            target.outputStream().use { it.write(bytes) }
                            bytes.size.toLong()
                        }
                    } ?: 0L
                    if (copied > 0L) target else {
                        target.delete()
                        null
                    }
                }
                if (file == null) {
                    _searchState.value = PhotosSearchState.Error(
                        "Couldn't read that photo. Use an image up to 8 MB and try again.",
                    )
                    return@launch
                }
                cacheFile = file
                searchBySelfie(file)
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                _searchState.value = PhotosSearchState.Error(e.localizedMessage ?: "Failed to process the selfie image.")
            } finally {
                cacheFile?.delete()
            }
        }
    }

    /**
     * Face search with a SPECIFIC stored selfie — the cockpit's picker grid
     * (web SelfieSearchPanel parity: tapping a thumbnail fires the search).
     */
    fun searchBySelfieId(selfieId: String) {
        val event = _activeEvent.value ?: return
        _isFiltered.value = true
        isFaceSearchMode = true
        val token = SessionManager.getInstance(getApplication()).getAccessToken()
        if (token == null) {
            _searchState.value = PhotosSearchState.Error("Authentication token not found. Please log in.")
            return
        }
        viewModelScope.launch {
            _searchState.value = PhotosSearchState.Loading
            try {
                val response = RetrofitClient.apiService.searchPhotosByFaceJson(
                    token = "Bearer $token",
                    slug = event.slug,
                    request = SearchByFaceJsonRequest(selfieId = selfieId)
                )
                if (response.success && response.data != null) {
                    _searchState.value = PhotosSearchState.Success(
                        photos = response.data.items,
                        total = response.data.items.size.toLong(),
                    )
                } else {
                    val err = response.errors?.firstOrNull()
                    _searchState.value =
                        PhotosSearchState.Error(humanizeFaceSearchError(err?.code, err?.message))
                }
            } catch (e: Exception) {
                _searchState.value = PhotosSearchState.Error(faceSearchErrorFrom(e))
            }
        }
    }

    fun searchByStoredSelfie() {
        val cached = _selfies.value.firstOrNull { it.isPrimary }
        if (cached != null) {
            searchBySelfieId(cached.id)
            return
        }
        // The CTA that lands here renders exactly when the cached library is
        // empty (still loading, or the fetch failed) — re-fetch before
        // concluding the runner has no primary selfie.
        val token = SessionManager.getInstance(getApplication()).getAccessToken()
        if (token == null) {
            _searchState.value = PhotosSearchState.Error("Authentication token not found. Please log in.")
            return
        }
        viewModelScope.launch {
            val primary = try {
                val response = RetrofitClient.apiService.getSelfies("Bearer $token")
                if (response.success && response.data != null) _selfies.value = response.data
                _selfies.value.firstOrNull { it.isPrimary }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                _searchState.value = PhotosSearchState.Error(faceSearchErrorFrom(e))
                return@launch
            }
            if (primary == null) {
                _searchState.value = PhotosSearchState.Error(
                    "No primary selfie set. Please upload a selfie and set it as primary first.",
                )
                return@launch
            }
            searchBySelfieId(primary.id)
        }
    }

    fun togglePhotoAlert(slug: String, register: Boolean) {
        val current = _photoAlert.value as? PhotoAlertUiState.Ready ?: return
        viewModelScope.launch {
            _photoAlert.value = current.copy(updating = true, message = null)
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
                        if (resp.success) PhotoAlertUiState.Ready(registered = true)
                        else current.copy(
                            updating = false,
                            message = resp.error ?: "Couldn't turn on photo alerts. Try again.",
                        )
                } else {
                    RetrofitClient.apiService.unregisterPhotoAlert("Bearer $token", slug)
                    _photoAlert.value = PhotoAlertUiState.Ready(registered = false)
                }
            } catch (e: Exception) {
                // Discriminate on the machine-readable code, not the message
                // text: SELFIE_REQUIRED means the selfie vanished between load
                // and toggle — route them to add one. Everything else (e.g.
                // EVENT_NOT_UPLOADABLE once the alert window closes) reverts
                // WITH the server's reason shown, never silently.
                val err = RetrofitClient.parseHttpError(e)
                _photoAlert.value = when (err?.code) {
                    "SELFIE_REQUIRED" -> PhotoAlertUiState.NeedsSelfie
                    else -> current.copy(
                        updating = false,
                        message = err?.message ?: RetrofitClient.parseError(e),
                    )
                }
            }
        }
    }

    private companion object {
        // Long enough to collapse a burst upload into one refetch, short enough
        // that a single shot still feels immediate.
        const val LIVE_REFRESH_DEBOUNCE_MS = 1_500L

        // Keystroke debounce for the bib field — collapses "1234" into one
        // request without making a submitted search feel laggy.
        const val BIB_SEARCH_DEBOUNCE_MS = 350L
    }
}
