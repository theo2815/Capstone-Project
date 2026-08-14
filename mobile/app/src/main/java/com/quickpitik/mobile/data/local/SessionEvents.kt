package com.quickpitik.mobile.data.local

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

// One-shot signal raised when the 401-refresh path gives up (no refresh token,
// or the backend rejected it). MainActivity collects this and bounces to login,
// so an expired session lands on the login screen instead of leaving the user
// staring at a surface whose every request silently 401s.
//
// extraBufferCapacity=1 keeps tryEmit() non-suspending, which matters because
// TokenAuthenticator raises it from OkHttp's thread, outside any coroutine.
object SessionEvents {
    private val _forcedLogout = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
    val forcedLogout: SharedFlow<Unit> = _forcedLogout

    fun raiseForcedLogout() {
        _forcedLogout.tryEmit(Unit)
    }
}
