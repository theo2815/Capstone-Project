package com.quickpitik.mobile.data.local

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

// One-shot signal raised when the photographer taps "Stop" on the tether
// ingest notification. TetherIngestService stops a live watch directly through
// CameraIngestCoordinator, then raises this for TetherViewModel to close a manual
// card import; its live-watch check is harmless after the service stop.
//
// Same shape as SessionEvents (extraBufferCapacity=1 keeps tryEmit()
// non-suspending) because the service raises it from onStartCommand, which
// runs on the main thread outside any coroutine.
object TetherEvents {
    private val _stopRequested = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
    val stopRequested: SharedFlow<Unit> = _stopRequested

    fun raiseStopRequested() {
        _stopRequested.tryEmit(Unit)
    }
}
