package com.quickpitik.mobile.data.local

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

// One-shot signal raised when the photographer taps "Stop" on the tether
// ingest notification. TetherIngestService is a dumb lifeline — it has no
// reference to the controllers or the ViewModel that own the USB loop — so it
// raises this and PhotographerDashboardViewModel decides which flow to end.
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
