package com.quickpitik.mobile.data.usb.fuji

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import com.quickpitik.mobile.BuildConfig
import com.quickpitik.mobile.data.usb.ptp.AndroidPtpTransport
import com.quickpitik.mobile.data.usb.ptp.CameraSetupException
import com.quickpitik.mobile.data.usb.ptp.LiveCamera
import com.quickpitik.mobile.data.usb.ptp.PtpDeviceInfo
import com.quickpitik.mobile.data.usb.ptp.PtpException
import com.quickpitik.mobile.data.usb.ptp.fingerprint

/** An opened USB device as the SDK needs it; the connection is closed through [close]. */
class UsbHandle(val fileDescriptor: Int, val rawDescriptors: ByteArray, val close: () -> Unit)

object FujiSdkCameras {
    fun androidHandle(manager: UsbManager): (UsbDevice) -> UsbHandle? = { device ->
        manager.openDevice(device)?.let { UsbHandle(it.fileDescriptor, it.rawDescriptors) { it.close() } }
    }
    internal fun ownershipAvailable() = AndroidPtpTransport.ownership.availablePermits() == 1

    /**
     * Null means "not for this build or device, use PTP". A [CameraSetupException] means the body
     * needs a menu change. The SDK owns the USB interface until the returned camera is closed.
     */
    fun open(
        device: UsbDevice,
        connect: (UsbDevice) -> UsbHandle?,
        api: FujiSdkApi = FujiSdkNative,
        enabled: Boolean = BuildConfig.FUJI_SDK && Build.VERSION.SDK_INT >= 30,
        vendorId: Int = device.vendorId,
    ): LiveCamera? {
        if (!enabled || vendorId != 0x04cb) return null
        if (!AndroidPtpTransport.ownership.tryAcquire()) throw PtpException("Camera is finishing another transfer. Try again shortly.")
        var owned = true
        var usb: UsbHandle? = null
        var started = false
        try {
            api.load()
            api.init()
            started = true
            usb = connect(device) ?: throw PtpException("USB permission is required.")
            val handle = try {
                api.open(usb.fileDescriptor, usb.rawDescriptors)
            } catch (e: FujiSdkException) {
                when (e.code) {
                    FujiSdk.ERR_INVALID_USBMODE -> throw CameraSetupException(
                        "Set the camera's PC CONNECTION MODE to USB TETHER SHOOTING AUTO, then start again. Use USB CARD READER only for import.")
                    FujiSdk.ERR_NO_MODEL_MODULE -> return null // the SDK has no module for this body: standard PTP path
                    else -> throw e
                }
            }
            val fuji = api.deviceInfo(handle)
            val info = PtpDeviceInfo(100, 0x0E, 0, "fujifilm.co.jp: 1.0;", emptyList(), emptyList(), emptyList(), emptyList(),
                listOf(0x3801), fuji.vendor.ifBlank { "FUJIFILM" }, fuji.product, fuji.firmware, fuji.serial)
            val key = fingerprint(vendorId, device.productId, info.manufacturer, info.model,
                fuji.serial.ifBlank { "attachment:${device.deviceName}:${device.deviceId}" })
            val handoff = usb
            val camera = FujiSdkCamera(api, handle, info, key) { handoff.close(); AndroidPtpTransport.ownership.release() }
            // From here the camera owns the SDK session, the USB connection and the semaphore.
            owned = false; usb = null; started = false
            try { camera.start() } catch (e: Throwable) { camera.close(); throw e }
            return camera
        } finally {
            if (started) runCatching { api.exit() }
            usb?.let { runCatching { it.close() } }
            if (owned) AndroidPtpTransport.ownership.release()
        }
    }
}
