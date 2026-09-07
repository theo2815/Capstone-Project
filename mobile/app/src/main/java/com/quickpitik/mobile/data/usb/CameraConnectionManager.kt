package com.quickpitik.mobile.data.usb

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import com.quickpitik.mobile.data.usb.ptp.UsbPtpInterfaces
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

sealed class CameraConnectionState {
    object Disconnected : CameraConnectionState()
    object Searching : CameraConnectionState()
    data class PermissionRequired(val deviceName: String, val denied: Boolean = false) : CameraConnectionState()
    data class Unavailable(val message: String) : CameraConnectionState()
    /** Attached and permission granted. PTP identification happens when an operation starts. */
    data class Connected(val deviceName: String, val vendorId: Int, val productId: Int) : CameraConnectionState()
}

class CameraConnectionManager(context: Context) {
    private val appContext = context.applicationContext
    private val usbManager = appContext.getSystemService(Context.USB_SERVICE) as UsbManager
    private val _state = MutableStateFlow<CameraConnectionState>(CameraConnectionState.Disconnected)
    val state: StateFlow<CameraConnectionState> = _state.asStateFlow()
    private var registered = false
    @Volatile private var selected: UsbDevice? = null
    private var generation = 0L
    private var permissionPending = false
    private var denied = false

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            @Suppress("DEPRECATION")
            val device = intent?.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)
            when (intent?.action) {
                UsbManager.ACTION_USB_DEVICE_ATTACHED -> refresh()
                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    if (device?.deviceName == selected?.deviceName && device?.deviceId == selected?.deviceId) clearSelection()
                    refresh(requestPermission = false)
                }
                ACTION_USB_PERMISSION -> {
                    if (device?.deviceName != selected?.deviceName ||
                        device?.deviceId != selected?.deviceId || intent.getLongExtra("generation", -1) != generation) return
                    permissionPending = false
                    denied = !usbManager.hasPermission(device ?: return)
                    refresh(requestPermission = false)
                }
            }
        }
    }

    fun start() {
        if (!registered) {
            val filter = IntentFilter().apply {
                addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
                addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
                addAction(ACTION_USB_PERMISSION)
            }
            ContextCompat.registerReceiver(appContext, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
            registered = true
        }
        refresh()
    }

    fun stop() {
        if (registered) runCatching { appContext.unregisterReceiver(receiver) }
        registered = false
    }

    fun refresh(requestPermission: Boolean = true) {
        if (!appContext.packageManager.hasSystemFeature(PackageManager.FEATURE_USB_HOST)) {
            _state.value = CameraConnectionState.Unavailable("This phone does not report USB host support.")
            return
        }
        val devices = usbManager.deviceList.values
        val camera = selected?.let { old -> devices.firstOrNull { it.deviceName == old.deviceName && it.deviceId == old.deviceId } }
            ?: devices.filter(::isCamera).sortedBy { it.deviceName }.firstOrNull()
        if (camera == null) {
            clearSelection()
            _state.value = if (devices.any { it.vendorId in setOf(0x04a9, 0x04b0, 0x054c, 0x04cb, 0x04da) })
                CameraConnectionState.Unavailable("The attached device has no usable PTP USB interface. Select PTP/MTP or a supported PC tether mode; storage and webcam modes cannot transfer photos here.")
                else CameraConnectionState.Disconnected
            return
        }
        if (selected?.deviceName != camera.deviceName || selected?.deviceId != camera.deviceId) {
            clearSelection(); selected = camera
        }
        val name = displayName(camera)
        if (usbManager.hasPermission(camera)) {
            permissionPending = false; denied = false
            _state.value = CameraConnectionState.Connected(name, camera.vendorId, camera.productId)
        } else {
            _state.value = CameraConnectionState.PermissionRequired(name, denied)
            if (requestPermission && !permissionPending) {
                denied = false; permissionPending = true
                generation++
                _state.value = CameraConnectionState.PermissionRequired(name)
                val intent = Intent(ACTION_USB_PERMISSION).setPackage(appContext.packageName)
                    .putExtra("generation", generation)
                val flags = PendingIntent.FLAG_UPDATE_CURRENT or
                    if (android.os.Build.VERSION.SDK_INT >= 31) PendingIntent.FLAG_MUTABLE else 0
                try { usbManager.requestPermission(camera, PendingIntent.getBroadcast(appContext, generation.toInt(), intent, flags)) }
                catch (e: RuntimeException) {
                    permissionPending = false; denied = true
                    _state.value = CameraConnectionState.PermissionRequired(name, true)
                }
            }
        }
    }

    private fun clearSelection() { selected = null; generation++; permissionPending = false; denied = false }
    private fun isCamera(device: UsbDevice) = UsbPtpInterfaces.find(device) != null
    private fun displayName(device: UsbDevice) = runCatching {
        listOfNotNull(device.manufacturerName, device.productName).distinct().joinToString(" ").ifBlank { "USB camera" }
    }.getOrDefault("USB camera")

    fun connectedCameraDevice(): UsbDevice? {
        val chosen = selected ?: return null
        return usbManager.deviceList[chosen.deviceName]?.takeIf {
            it.deviceId == chosen.deviceId && usbManager.hasPermission(it)
        }
    }
    val manager: UsbManager get() = usbManager
    companion object { internal const val ACTION_USB_PERMISSION = "com.quickpitik.mobile.USB_PERMISSION" }
}
