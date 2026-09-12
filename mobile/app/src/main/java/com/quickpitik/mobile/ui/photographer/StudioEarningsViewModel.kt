package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.EarningsOverviewDto
import com.quickpitik.mobile.data.remote.PayoutBalanceDto
import com.quickpitik.mobile.data.remote.PhotographerTransactionDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
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

/**
 * The photographer's wallet: the earnings overview, the payout balance, the
 * transactions ledger, and the two payout-cycle actions (request, withdraw).
 *
 * Split out of [PhotographerDashboardViewModel] (plan 2 phase G). Its `init`
 * loads only the three earnings routes, so a payout test stubs those rather
 * than the dashboard's whole cold-start salvo.
 *
 * Scoped to the `studio` nav graph like the ViewModels beside it.
 */
class StudioEarningsViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)

    private val _earningsUiState = MutableStateFlow<EarningsUiState>(EarningsUiState.Loading)
    val earningsUiState: StateFlow<EarningsUiState> = _earningsUiState

    // ponytail: "processing" is a sentinel inside a String? channel, the same
    // smell ActionState replaced on the settings side. Left as-is because the
    // earnings screen reads it in three places and phase G's brief named only
    // the settings channel; convert it with the next earnings-screen change.
    private val _payoutActionState = MutableStateFlow<String?>(null)
    val payoutActionState: StateFlow<String?> = _payoutActionState

    init {
        fetchEarningsAndTransactions()
    }

    fun fetchEarningsAndTransactions() {
        viewModelScope.launch {
            _earningsUiState.value = EarningsUiState.Loading
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _earningsUiState.value = EarningsUiState.Error("No valid session. Please log in again.")
                return@launch
            }

            try {
                // Independent endpoints — fired concurrently. Sequential
                // awaits made the Earnings tab pay the sum of three
                // round-trips instead of the slowest one.
                val (overviewResponse, balanceResponse, transactionsResponse) =
                    coroutineScope {
                        val overview = async {
                            RetrofitClient.apiService.getEarningsOverview("Bearer $token")
                        }
                        val balance = async {
                            RetrofitClient.apiService.getPayoutBalance("Bearer $token")
                        }
                        val transactions = async {
                            RetrofitClient.apiService.getTransactionsLedger("Bearer $token")
                        }
                        Triple(overview.await(), balance.await(), transactions.await())
                    }

                if (overviewResponse.success && overviewResponse.data != null &&
                    balanceResponse.success && balanceResponse.data != null &&
                    transactionsResponse.success && transactionsResponse.data != null
                ) {
                    _earningsUiState.value = EarningsUiState.Success(
                        overview = overviewResponse.data,
                        balance = balanceResponse.data,
                        transactions = transactionsResponse.data.items,
                        monthTotals = transactionsResponse.data.monthTotals,
                    )
                } else {
                    val errMsg = overviewResponse.error
                        ?: balanceResponse.error
                        ?: transactionsResponse.error
                        ?: "Failed to load financial records."
                    _earningsUiState.value = EarningsUiState.Error(errMsg)
                }
            } catch (e: Exception) {
                _earningsUiState.value = EarningsUiState.Error(RetrofitClient.parseError(e))
            }
        }
    }

    fun submitPayoutRequest() {
        viewModelScope.launch {
            _payoutActionState.value = "processing"
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _payoutActionState.value = "Error: No active login session."
                return@launch
            }

            try {
                val response = RetrofitClient.apiService.requestPayout("Bearer $token")
                if (response.success && response.data != null) {
                    _payoutActionState.value = "Success: Payout request submitted successfully!"
                    fetchEarningsAndTransactions()
                } else {
                    _payoutActionState.value = "Error: ${response.error ?: "Payout request rejected."}"
                }
            } catch (e: Exception) {
                // PAYOUT_NO_ACCOUNT / PAYOUT_BELOW_MINIMUM / PAYOUT_REQUEST_OPEN
                // arrive as non-2xx — surface the server's reason, not
                // "HTTP 409 Conflict".
                val err = RetrofitClient.parseHttpError(e)
                _payoutActionState.value =
                    "Error: ${err?.message ?: RetrofitClient.parseError(e)}"
            }
        }
    }

    fun clearPayoutActionState() {
        _payoutActionState.value = null
    }

    /** Withdraws a HELD payout request (web WithdrawPayoutModal parity). */
    fun withdrawPayoutRequest(cycleId: String) {
        viewModelScope.launch {
            _payoutActionState.value = "processing"
            val token = sessionManager.getAccessToken()
            if (token == null) {
                _payoutActionState.value = "Error: No active login session."
                return@launch
            }
            try {
                val response = RetrofitClient.apiService.withdrawPayout("Bearer $token", cycleId)
                if (response.success) {
                    _payoutActionState.value =
                        "Success: Request withdrawn — you can file a new one when ready."
                    fetchEarningsAndTransactions()
                } else {
                    _payoutActionState.value =
                        "Error: ${response.error ?: "Couldn't withdraw the request."}"
                }
            } catch (e: Exception) {
                val err = RetrofitClient.parseHttpError(e)
                _payoutActionState.value =
                    "Error: ${err?.message ?: RetrofitClient.parseError(e)}"
            }
        }
    }
}
