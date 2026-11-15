package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.UserDto
import com.quickpitik.mobile.data.repository.ProfileRepository
import com.quickpitik.mobile.data.repository.ProfileRepositoryImpl
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/** The prompt is due only when the backend asks for it and names the version to accept. */
internal fun mustAcceptTerms(user: UserDto): Boolean =
    user.termsAcceptanceRequired && !user.legalCurrentVersion.isNullOrBlank()

data class TermsState(
    val required: Boolean = false,
    val version: String? = null,
    val busy: Boolean = false,
    val error: String? = null,
)

/**
 * RV-L6: the blocking "accept the updated Terms" prompt of the studio shell.
 * The app has no Terms version of its own; it accepts the one the backend named.
 * A failed load shows no prompt: the backend withholds the automatic actions until
 * the photographer accepts, so nothing is lost by staying quiet.
 */
class TermsAcceptanceViewModel @JvmOverloads constructor(
    application: Application,
    private val repository: ProfileRepository = ProfileRepositoryImpl(),
) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val _state = MutableStateFlow(TermsState())
    val state: StateFlow<TermsState> = _state

    /** Asks the backend again; the studio calls it each time it comes to the foreground, so a failed first load or a new version is picked up. */
    fun load() {
        val token = sessionManager.getAccessToken() ?: return
        if (_state.value.busy) return
        viewModelScope.launch { repository.getMe(token).onSuccess { apply(it) } }
    }

    fun accept() {
        val token = sessionManager.getAccessToken() ?: return
        val current = _state.value
        val version = current.version ?: return
        if (current.busy) return
        _state.value = current.copy(busy = true, error = null)
        viewModelScope.launch {
            repository.acceptTerms(token, version)
                .onSuccess { apply(it) }
                .onFailure { err ->
                    _state.value = _state.value.copy(busy = false, error = err.message)
                    // A refusal usually means the version moved: fetch the one the backend names now, so the next tap posts it.
                    repository.getMe(token).onSuccess { apply(it, err.message) }
                }
        }
    }

    private fun apply(user: UserDto, error: String? = null) {
        _state.value = TermsState(required = mustAcceptTerms(user), version = user.legalCurrentVersion, error = error)
    }
}
