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
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import com.quickpitik.mobile.MainActivity
import com.quickpitik.mobile.data.local.TetherEvents
import com.quickpitik.mobile.data.local.UploadSpool
import com.quickpitik.mobile.data.usb.CameraIngestCoordinator
import com.quickpitik.mobile.data.usb.CameraConnectionState
import com.quickpitik.mobile.data.usb.ShutterWatchState
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.combine

/** Owns camera coroutine lifetime, the notification and wake lock. */
class TetherIngestService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val coordinator by lazy { CameraIngestCoordinator.get(this) }
    private var foreground = false
    private var lastNotice: String? = null
    private var storageAlert: UploadSpool.Level? = null

    override fun onCreate() {
        super.onCreate()
        coordinator.attach()
        scope.launch {
            combine(coordinator.watch, coordinator.cardBusy, coordinator.cameraManager.state,
                UploadSpool.observeSpace(this@TetherIngestService)) { state, busy, connection, space ->
                Triple(state, busy, connection) to space
            }.collect { (capture, space) ->
                val (state, busy, connection) = capture
                if (foreground) {
                    showStorageAlert(space)
                    if (connection is CameraConnectionState.Connected && (coordinator.isWatching || busy)) acquireWakeLock()
                    else releaseWakeLock()
                    val notice = when (state) {
                        is ShutterWatchState.Watching -> state.pauseReason ?: if (state.reconnecting)
                            "Reconnecting to camera..." else "${state.captureCount} photos saved on phone"
                        ShutterWatchState.Starting -> "Starting camera..."
                        ShutterWatchState.Stopping -> "Finishing camera transfer..."
                        else -> null
                    }
                    if (notice != null) showNotice(notice)
                    else if (!busy) stopSelf()
                }
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                coordinator.stopWatch()
                TetherEvents.raiseStopRequested()
                if (!coordinator.isWatching && !coordinator.cardBusy.value) stopSelf()
            }
            ACTION_START, ACTION_UPDATE, ACTION_WATCH -> {
                val notice = intent.getStringExtra(EXTRA_NOTICE) ?: DEFAULT_NOTICE
                try {
                    showNotice(notice)
                    if (coordinator.cameraManager.state.value is CameraConnectionState.Connected) acquireWakeLock()
                    if (intent.action == ACTION_START)
                        coordinator.beginCard(scope, intent.getLongExtra("card_generation", -1))
                    if (intent.action == ACTION_WATCH) {
                        val owner = intent.getStringExtra("owner") ?: return START_NOT_STICKY
                        val event = intent.getStringExtra("event") ?: return START_NOT_STICKY
                        coordinator.beginWatch(scope, owner, event, intent.getLongExtra("generation", -1))
                    }
                    if (!coordinator.isWatching && !coordinator.cardBusy.value) stopSelf()
                } catch (e: RuntimeException) {
                    coordinator.startFailed()
                    stopSelf()
                }
            }
        }
        return START_NOT_STICKY
    }

    private fun showNotice(notice: String) {
        if (foreground && lastNotice == notice) return
        ensureChannel()
        if (!foreground) {
            ServiceCompat.startForeground(this, NOTIFICATION_ID, buildNotification(notice),
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)
            foreground = true
        } else {
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
                .notify(NOTIFICATION_ID, buildNotification(notice))
        }
        lastNotice = notice
    }

    private fun showStorageAlert(space: UploadSpool.Space) {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (space.level == UploadSpool.Level.READY) {
            manager.cancel(STORAGE_NOTIFICATION_ID)
            storageAlert = null
            return
        }
        if (storageAlert == space.level || (storageAlert == UploadSpool.Level.BLOCKED && space.level == UploadSpool.Level.LOW)) return
        if (!NotificationManagerCompat.from(this).areNotificationsEnabled()) return
        manager.createNotificationChannel(NotificationChannel(STORAGE_CHANNEL_ID, "Phone storage", NotificationManager.IMPORTANCE_DEFAULT))
        val title = when (space.level) {
            UploadSpool.Level.BLOCKED -> "Phone storage needs attention"
            UploadSpool.Level.UNKNOWN -> "Couldn't check phone storage"
            else -> "Phone storage is getting low"
        }
        val detail = "Open Capture to free verified phone copies. Camera transfers pause when space is insufficient; uploads can continue."
        val open = PendingIntent.getActivity(this, REQUEST_CONTENT, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        manager.cancel(STORAGE_NOTIFICATION_ID)
        manager.notify(STORAGE_NOTIFICATION_ID, NotificationCompat.Builder(this, STORAGE_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_warning)
            .setContentTitle(title).setContentText(detail)
            .setStyle(NotificationCompat.BigTextStyle().bigText(detail))
            .setContentIntent(open).setAutoCancel(true)
            .setCategory(NotificationCompat.CATEGORY_ERROR).build())
        storageAlert = space.level
    }

    override fun onDestroy() {
        coordinator.detach()
        scope.cancel()
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
        const val ACTION_WATCH = "com.quickpitik.mobile.tether.WATCH"
        const val ACTION_START = "com.quickpitik.mobile.tether.START"
        const val ACTION_UPDATE = "com.quickpitik.mobile.tether.UPDATE"
        const val ACTION_STOP = "com.quickpitik.mobile.tether.STOP"
        const val EXTRA_NOTICE = "notice"

        private const val CHANNEL_ID = "tether_ingest"
        private const val CHANNEL_NAME = "Tethered shoot"
        private const val CHANNEL_DESCRIPTION =
            "Shows while photos are transferring from a connected camera."
        private const val NOTIFICATION_ID = 4101
        private const val STORAGE_NOTIFICATION_ID = 4102
        private const val STORAGE_CHANNEL_ID = "tether_storage"
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
