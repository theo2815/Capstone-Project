package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.isPhotographerRole
import com.quickpitik.mobile.data.remote.EarningsOverviewDto
import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PerEventEarningDto
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.launch

sealed class EarningsUiState {
    object Loading : EarningsUiState()
    data class Success(
        val overview: EarningsOverviewDto,
        val balance: PayoutBalanceDto,
        val transactions: List<PhotographerTransactionDto>,
        // Server-computed per-month totals over ALL transactions (not just the
        // loaded page) — drives the ledger's month subtotals.
        val monthTotals: Map<String, Double> = emptyMap(),
    ) : EarningsUiState()
    data class Error(val message: String) : EarningsUiState()
}

data class TransactionsPageState(
    val items: List<PhotographerTransactionDto> = emptyList(),
    val total: Long = 0,
    val nextOffset: Int = 0,
    val isLoading: Boolean = false,
    val isLoadingMore: Boolean = false,
    val error: String? = null,
    val appendError: String? = null,
    val hasMore: Boolean = true,
    val monthTotals: Map<String, Double> = emptyMap(),
)

data class EventEarningsPageState(
    val items: List<PerEventEarningDto> = emptyList(),
    val total: Long = 0,
    val nextOffset: Int = 0,
    val isLoading: Boolean = false,
    val isLoadingMore: Boolean = false,
    val error: String? = null,
    val appendError: String? = null,
    val hasMore: Boolean = true,
)

/** The photographer's wallet, ledger, per-event earnings, and payout actions. */
class StudioEarningsViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val _earningsUiState = MutableStateFlow<EarningsUiState>(EarningsUiState.Loading)
    val earningsUiState: StateFlow<EarningsUiState> = _earningsUiState
    private val _transactionsState = MutableStateFlow(TransactionsPageState())
    val transactionsState: StateFlow<TransactionsPageState> = _transactionsState
    private val _eventEarningsState = MutableStateFlow(EventEarningsPageState())
    val eventEarningsState: StateFlow<EventEarningsPageState> = _eventEarningsState
    // ponytail: "processing" remains a sentinel in this existing String? channel;
    // the earnings screen still reads it. Change it with that screen's next edit.
    private val _payoutActionState = MutableStateFlow<String?>(null)
    val payoutActionState: StateFlow<String?> = _payoutActionState

    private var walletJob: Job? = null
    private var transactionsJob: Job? = null
    private var eventsJob: Job? = null
    private var payoutJob: Job? = null
    private var walletRequest = 0L
    private var transactionsRequest = 0L
    private var eventsRequest = 0L
    private var payoutRequest = 0L

    init {
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            sessionManager.authChanges.drop(1).collect {
                walletRequest++
                transactionsRequest++
                eventsRequest++
                payoutRequest++
                walletJob?.cancel()
                transactionsJob?.cancel()
                eventsJob?.cancel()
                payoutJob?.cancel()
                _earningsUiState.value = EarningsUiState.Loading
                _transactionsState.value = TransactionsPageState(hasMore = false)
                _eventEarningsState.value = EventEarningsPageState(hasMore = false)
                _payoutActionState.value = null
            }
        }
        fetchEarningsAndTransactions()
    }

    fun fetchEarningsAndTransactions() {
        walletRequest++
        walletJob?.cancel()
        _earningsUiState.value = EarningsUiState.Loading
        val request = walletRequest
        val generation = sessionManager.authChanges.value
        val token = sessionManager.getAccessToken()?.takeIf { isPhotographerRole(sessionManager.getUserRole()) }
        if (token == null) {
            _earningsUiState.value = EarningsUiState.Error("No valid session. Please log in again.")
        } else walletJob = viewModelScope.launch {
            try {
                val (overview, balance) = coroutineScope {
                    val first = async { RetrofitClient.apiService.getEarningsOverview("Bearer $token") }
                    val second = async { RetrofitClient.apiService.getPayoutBalance("Bearer $token") }
                    first.await() to second.await()
                }
                if (request != walletRequest || generation != sessionManager.authChanges.value) return@launch
                _earningsUiState.value = if (overview.success && overview.data != null && balance.success && balance.data != null)
                    EarningsUiState.Success(overview.data, balance.data, _transactionsState.value.items, _transactionsState.value.monthTotals)
                else EarningsUiState.Error(overview.error ?: balance.error ?: "Failed to load financial records.")
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (request == walletRequest && generation == sessionManager.authChanges.value)
                    _earningsUiState.value = EarningsUiState.Error(RetrofitClient.parseError(e))
            }
        }
        loadTransactions(refresh = true)
        loadEventEarnings(refresh = true)
    }

    fun loadMoreTransactions() = loadTransactions(refresh = false)
    fun retryTransactions() = loadTransactions(refresh = _transactionsState.value.appendError == null)
    fun loadMoreEventEarnings() = loadEventEarnings(refresh = false)
    fun retryEventEarnings() = loadEventEarnings(refresh = _eventEarningsState.value.appendError == null)

    private fun loadTransactions(refresh: Boolean) {
        val current = _transactionsState.value
        if (!refresh && (current.isLoading || current.isLoadingMore || !current.hasMore)) return
        if (refresh) { transactionsRequest++; transactionsJob?.cancel() }
        val request = transactionsRequest
        val generation = sessionManager.authChanges.value
        val token = sessionManager.getAccessToken()?.takeIf { isPhotographerRole(sessionManager.getUserRole()) } ?: return
        val offset = if (refresh) 0 else current.nextOffset
        _transactionsState.value = if (refresh) TransactionsPageState(isLoading = true)
            else current.copy(isLoadingMore = true, appendError = null)
        transactionsJob = viewModelScope.launch {
            try {
                val response = RetrofitClient.apiService.getTransactionsLedger("Bearer $token", offset, 50)
                if (request != transactionsRequest || generation != sessionManager.authChanges.value) return@launch
                val page = response.data.takeIf { response.success } ?: error(response.error ?: "Couldn't load transactions.")
                val previous = if (refresh) emptyList() else _transactionsState.value.items
                val next = page.offset + page.limit
                val items = (previous + page.items).distinctBy { it.id }
                _transactionsState.value = TransactionsPageState(items, page.total, next, hasMore = page.items.isNotEmpty() && next < page.total, monthTotals = page.monthTotals)
                val wallet = _earningsUiState.value
                if (wallet is EarningsUiState.Success) _earningsUiState.value = wallet.copy(transactions = items, monthTotals = page.monthTotals)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (request != transactionsRequest || generation != sessionManager.authChanges.value) return@launch
                val message = RetrofitClient.parseHttpError(e)?.message ?: RetrofitClient.parseError(e)
                _transactionsState.value = if (refresh) TransactionsPageState(error = message)
                    else _transactionsState.value.copy(isLoadingMore = false, appendError = message)
            }
        }
    }

    private fun loadEventEarnings(refresh: Boolean) {
        val current = _eventEarningsState.value
        if (!refresh && (current.isLoading || current.isLoadingMore || !current.hasMore)) return
        if (refresh) { eventsRequest++; eventsJob?.cancel() }
        val request = eventsRequest
        val generation = sessionManager.authChanges.value
        val token = sessionManager.getAccessToken()?.takeIf { isPhotographerRole(sessionManager.getUserRole()) } ?: return
        val offset = if (refresh) 0 else current.nextOffset
        _eventEarningsState.value = if (refresh) EventEarningsPageState(isLoading = true)
            else current.copy(isLoadingMore = true, appendError = null)
        eventsJob = viewModelScope.launch {
            try {
                val response = RetrofitClient.apiService.getPhotographerPerEventEarnings("Bearer $token", offset, 10)
                if (request != eventsRequest || generation != sessionManager.authChanges.value) return@launch
                val page = response.data.takeIf { response.success } ?: error(response.error ?: "Couldn't load event earnings.")
                val previous = if (refresh) emptyList() else _eventEarningsState.value.items
                val next = page.offset + page.limit
                _eventEarningsState.value = EventEarningsPageState((previous + page.items).distinctBy { it.eventId }, page.total, next, hasMore = page.items.isNotEmpty() && next < page.total)
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (request != eventsRequest || generation != sessionManager.authChanges.value) return@launch
                val message = RetrofitClient.parseHttpError(e)?.message ?: RetrofitClient.parseError(e)
                _eventEarningsState.value = if (refresh) EventEarningsPageState(error = message)
                    else _eventEarningsState.value.copy(isLoadingMore = false, appendError = message)
            }
        }
    }

    fun submitPayoutRequest() {
        payoutRequest++
        payoutJob?.cancel()
        val request = payoutRequest
        val generation = sessionManager.authChanges.value
        payoutJob = viewModelScope.launch {
            _payoutActionState.value = "processing"
            val token = sessionManager.getAccessToken()?.takeIf { isPhotographerRole(sessionManager.getUserRole()) }
            if (token == null) {
                _payoutActionState.value = "Error: No active login session."
                return@launch
            }
            try {
                val response = RetrofitClient.apiService.requestPayout("Bearer $token")
                if (!currentPayout(request, generation)) return@launch
                if (response.success && response.data != null) {
                    _payoutActionState.value = "Success: Payout request submitted successfully!"
                    fetchEarningsAndTransactions()
                } else {
                    _payoutActionState.value = "Error: ${response.error ?: "Payout request rejected."}"
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (!currentPayout(request, generation)) return@launch
                val err = RetrofitClient.parseHttpError(e)
                _payoutActionState.value = "Error: ${err?.message ?: RetrofitClient.parseError(e)}"
            }
        }
    }

    fun clearPayoutActionState() { _payoutActionState.value = null }

    private fun currentPayout(request: Long, generation: Long): Boolean =
        request == payoutRequest && generation == sessionManager.authChanges.value &&
            isPhotographerRole(sessionManager.getUserRole()) && sessionManager.getAccessToken() != null

    fun withdrawPayoutRequest(cycleId: String) {
        payoutRequest++
        payoutJob?.cancel()
        val request = payoutRequest
        val generation = sessionManager.authChanges.value
        payoutJob = viewModelScope.launch {
            _payoutActionState.value = "processing"
            val token = sessionManager.getAccessToken()?.takeIf { isPhotographerRole(sessionManager.getUserRole()) }
            if (token == null) {
                _payoutActionState.value = "Error: No active login session."
                return@launch
            }
            try {
                val response = RetrofitClient.apiService.withdrawPayout("Bearer $token", cycleId)
                if (!currentPayout(request, generation)) return@launch
                if (response.success) {
                    _payoutActionState.value = "Success: Request withdrawn — you can file a new one when ready."
                    fetchEarningsAndTransactions()
                } else {
                    _payoutActionState.value = "Error: ${response.error ?: "Couldn't withdraw the request."}"
                }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) {
                if (!currentPayout(request, generation)) return@launch
                val err = RetrofitClient.parseHttpError(e)
                _payoutActionState.value = "Error: ${err?.message ?: RetrofitClient.parseError(e)}"
            }
        }
    }
}
