package com.quickpitik.mobile.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.quickpitik.mobile.MainActivity
import com.quickpitik.mobile.data.local.TetherEvents

/**
 * Keeps the process alive for the duration of a USB ingest — the live shutter
 * watch or a card import.
 *
 * **It does not own the ingest loop.** The loop stays in
 * `PhotographerDashboardViewModel.viewModelScope`, and that is deliberate: the
 * ViewModel is scoped to the `dashboard` nav entry, which is only popped on
 * logout, so it comfortably outlives a shoot. What actually kills a pocketed
 * phone's shoot is the *process* being cached — Android 12+ freezes cached
 * processes, so the coroutine is never cancelled, it simply stops being
 * scheduled. A `connectedDevice` foreground service is the documented fix for
 * exactly that, and it costs a fraction of the churn of relocating the loop.
 *
 * Three jobs, nothing more:
 *  1. hold the process in the foreground for as long as bytes are moving,
 *  2. hold a partial wakelock so the CPU doesn't suspend with the screen off
 *     (a foreground service alone does NOT keep the CPU awake),
 *  3. give the photographer a glanceable status and a Stop control in the shade.
 *
 * The caller supplies the notification text, so this class knows nothing about
 * cameras, handles or capture counts.
 */
class TetherIngestService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                // The shade's Stop button. The controllers live in the
                // ViewModel, so all we can do is say so and stand down; the VM
                // ends whichever flow is running, which drives the state that
                // would have stopped us anyway.
                TetherEvents.raiseStopRequested()
                stopSelf()
            }
            ACTION_START, ACTION_UPDATE -> {
                val notice = intent.getStringExtra(EXTRA_NOTICE) ?: DEFAULT_NOTICE
                ensureChannel()
                // Re-calling startForeground on an already-foreground service is
                // how the text gets updated — there is no separate update call.
                //
                // Android 14+ only permits the connectedDevice type while the
                // app holds a live USB device grant. The instant the cable is
                // pulled that grant is gone, and the very next notice update
                // ("reconnecting…") makes this call throw SecurityException —
                // which crashed the whole process mid-shoot (device-verified
                // 2026-09-02). The service is a lifeline, not the owner of the
                // ingest loop, so losing it must never take the loop down:
                // stand down quietly and let the next USB attach restart us.
                try {
                    ServiceCompat.startForeground(
                        this,
                        NOTIFICATION_ID,
                        buildNotification(notice),
                        ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE,
                    )
                } catch (e: RuntimeException) {
                    // SecurityException (no USB grant) or the API 31+
                    // ForegroundServiceStartNotAllowedException (backgrounded).
                    stopSelf()
                    return START_NOT_STICKY
                }
                acquireWakeLock()
            }
        }
        // NOT sticky. The coroutine that actually pulls frames lives in the
        // ViewModel, so a service the system restarts on its own would post a
        // notification claiming work that no longer exists.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG)
            .also { it.acquire(WAKE_LOCK_TIMEOUT_MS) }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    private fun ensureChannel() {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (manager.getNotificationChannel(CHANNEL_ID) != null) return
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                CHANNEL_NAME,
                // LOW: the text updates on every frame during a shoot. At
                // DEFAULT importance that would buzz the phone hundreds of
                // times over a race.
                NotificationManager.IMPORTANCE_LOW,
            ).apply {
                description = CHANNEL_DESCRIPTION
                setShowBadge(false)
            }
        )
    }

    private fun buildNotification(notice: String): Notification {
        val contentIntent = PendingIntent.getActivity(
            this,
            REQUEST_CONTENT,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val stopIntent = PendingIntent.getService(
            this,
            REQUEST_STOP,
            Intent(this, TetherIngestService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            // Framework icon rather than a new asset: this is a transfer in
            // progress, which is precisely what stat_sys_upload depicts.
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle(notice)
            .setContentText(SUBTEXT)
            .setContentIntent(contentIntent)
            .addAction(0, STOP_LABEL, stopIntent)
            .setOngoing(true)
            .setSilent(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    companion object {
        const val ACTION_START = "com.quickpitik.mobile.tether.START"
        const val ACTION_UPDATE = "com.quickpitik.mobile.tether.UPDATE"
        const val ACTION_STOP = "com.quickpitik.mobile.tether.STOP"
        const val EXTRA_NOTICE = "notice"

        private const val CHANNEL_ID = "tether_ingest"
        private const val CHANNEL_NAME = "Tethered shoot"
        private const val CHANNEL_DESCRIPTION =
            "Shows while photos are transferring from a connected camera."
        private const val NOTIFICATION_ID = 4101
        private const val REQUEST_CONTENT = 0
        private const val REQUEST_STOP = 1

        private const val DEFAULT_NOTICE = "Camera transfer in progress"
        private const val SUBTEXT = "Keep QuickPitik running until the shoot ends."
        private const val STOP_LABEL = "Stop"

        private const val WAKE_LOCK_TAG = "QuickPitik:tether-ingest"

        // A hard backstop, not an expected duration. The wakelock is released
        // when the service stops, which is what normally ends it; this only
        // guarantees that a bug which leaks the service can't drain the battery
        // indefinitely. Eight hours clears the longest realistic race day.
        private const val WAKE_LOCK_TIMEOUT_MS = 8L * 60L * 60L * 1000L
    }
}
