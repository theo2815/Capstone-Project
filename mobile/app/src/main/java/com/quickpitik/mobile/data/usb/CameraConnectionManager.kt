package com.quickpitik.mobile.data.usb

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Live state of the tethered camera link on the USB bus.
 *
 * A DSLR / mirrorless body set to "PC remote / PTP" enumerates as a USB
 * still-image-class device (class 6). We treat any device exposing that class
 * — on the device itself or one of its interfaces — as a camera.
 */
sealed class CameraConnectionState {
    /** Nothing camera-shaped is on the USB bus. */
    object Disconnected : CameraConnectionState()

    /** Transient — actively scanning the bus (screen entry / attach event). */
    object Searching : CameraConnectionState()

    /** A still-image-class device is attached. */
    data class Connected(
        val deviceName: String,
        val vendorId: Int,
        val productId: Int
    ) : CameraConnectionState()
}

/**
 * Real USB-host (USB-C OTG) detection of a tethered camera.
 *
 * **Scope (Build Mandate rule 4 — tether is the final milestone):** this detects
 * the camera *link only* — attach, detach, and USB permission. Pulling frames
 * off the body over PTP is the deferred final-milestone work; until it lands,
 * the Tether console drives a simulated capture stand-in once a real camera is
 * detected here. The screen contract (a [StateFlow] of [CameraConnectionState])
 * will not change when real PTP transfer is wired in — only this class's guts.
 */
class CameraConnectionManager(context: Context) {

    private val appContext = context.applicationContext
    private val usbManager =
        appContext.getSystemService(Context.USB_SERVICE) as UsbManager

    private val _state =
        MutableStateFlow<CameraConnectionState>(CameraConnectionState.Disconnected)
    val state: StateFlow<CameraConnectionState> = _state.asStateFlow()

    private var registered = false

    // Avoid re-prompting for permission on every refresh (the permission result
    // itself triggers a refresh). Keyed by the device's stable bus name.
    private var permissionAskedFor: String? = null

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context?, intent: Intent?) {
            when (intent?.action) {
                UsbManager.ACTION_USB_DEVICE_ATTACHED,
                UsbManager.ACTION_USB_DEVICE_DETACHED,
                ACTION_USB_PERMISSION -> refresh()
            }
        }
    }

    /** Start listening for bus changes and scan immediately. Idempotent. */
    fun start() {
        if (!registered) {
            val filter = IntentFilter().apply {
                addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
                addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
                addAction(ACTION_USB_PERMISSION)
            }
            // System USB broadcasts are exempt from the export restriction, so
            // NOT_EXPORTED is both safe and correct here.
            ContextCompat.registerReceiver(
                appContext, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED
            )
            registered = true
        }
        refresh()
    }

    /** Stop listening. Idempotent. */
    fun stop() {
        if (registered) {
            runCatching { appContext.unregisterReceiver(receiver) }
            registered = false
        }
    }

    /** Re-scan the bus and recompute [state]. Safe to call from the UI. */
    fun refresh() {
        _state.value = CameraConnectionState.Searching
        val camera = usbManager.deviceList.values.firstOrNull(::isCamera)
        _state.value = if (camera == null) {
            permissionAskedFor = null
            CameraConnectionState.Disconnected
        } else {
            ensurePermission(camera)
            CameraConnectionState.Connected(
                deviceName = displayName(camera),
                vendorId = camera.vendorId,
                productId = camera.productId
            )
        }
    }

    private fun isCamera(device: UsbDevice): Boolean {
        if (device.deviceClass == UsbConstants.USB_CLASS_STILL_IMAGE) return true
        for (i in 0 until device.interfaceCount) {
            if (device.getInterface(i).interfaceClass == UsbConstants.USB_CLASS_STILL_IMAGE) {
                return true
            }
        }
        return false
    }

    private fun displayName(device: UsbDevice): String {
        // productName / manufacturerName may be null until permission is granted;
        // a refresh fires when it is, at which point we re-read the real strings.
        val product = runCatching { device.productName }.getOrNull()
        val manufacturer = runCatching { device.manufacturerName }.getOrNull()
        return when {
            !product.isNullOrBlank() && !manufacturer.isNullOrBlank() -> "$manufacturer $product"
            !product.isNullOrBlank() -> product
            else -> "USB Camera"
        }
    }

    /** Request USB permission once per attached camera (needed for PTP later). */
    private fun ensurePermission(device: UsbDevice) {
        if (usbManager.hasPermission(device)) return
        if (permissionAskedFor == device.deviceName) return
        permissionAskedFor = device.deviceName

        val flags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
        } else {
            PendingIntent.FLAG_UPDATE_CURRENT
        }
        val intent = Intent(ACTION_USB_PERMISSION).setPackage(appContext.packageName)
        val pending = PendingIntent.getBroadcast(appContext, 0, intent, flags)
        usbManager.requestPermission(device, pending)
    }

    /**
     * The currently attached camera as a raw [UsbDevice] — but only if we already
     * hold USB permission for it (PTP needs an open connection). Returns null when
     * nothing camera-shaped is attached or the permission prompt hasn't been
     * accepted yet.
     */
    fun connectedCameraDevice(): UsbDevice? =
        usbManager.deviceList.values.firstOrNull(::isCamera)?.takeIf { usbManager.hasPermission(it) }

    /** System [UsbManager], so the PTP layer can open the device connection. */
    val manager: UsbManager get() = usbManager

    companion object {
        private const val ACTION_USB_PERMISSION = "com.quickpitik.mobile.USB_PERMISSION"
    }
}
