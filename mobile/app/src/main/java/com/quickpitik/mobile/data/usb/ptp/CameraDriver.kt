package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import java.io.Closeable
import java.io.OutputStream

data class CameraCapabilities(
    val cardImport: Boolean,
    val thumbnails: Boolean,
    val eventPolling: Boolean,
    // Unknown on generic PTP: actual physical-shutter use needs a hardware test.
    val physicalShutter: String,
    val imageFormats: List<Int>,
)

/** Protocol behavior only. Account/event persistence belongs above this boundary. */
open class GenericPtpCamera(
    val session: PtpSession,
    val info: PtpDeviceInfo,
    val cameraKey: String,
) : Closeable {
    open val capabilities = CameraCapabilities(
        cardImport = listOf(Ptp.OP_GET_STORAGE_IDS, Ptp.OP_GET_OBJECT_HANDLES, Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT)
            .all { it in info.operationsSupported },
        thumbnails = Ptp.OP_GET_THUMB in info.operationsSupported,
        eventPolling = false,
        physicalShutter = "Unverified; use card import if this USB mode locks the shutter.",
        imageFormats = info.imageFormats,
    )
    open fun startLive() {}
    open fun events(): List<Long> = emptyList()
    open fun keepAlive() {}
    open fun acknowledge(handle: Long) {}
    open fun handles(): List<Long> = session.getStorageIds().flatMap { session.getObjectHandles(it) }.distinct()
    open fun objectInfo(handle: Long) = session.getObjectInfo(handle)
    open fun readObject(handle: Long, sink: OutputStream, expectedSize: Long?) = session.getObjectTo(handle, sink, expectedSize)
    override fun close() = session.close()
}

class CanonEosCamera(session: PtpSession, info: PtpDeviceInfo, cameraKey: String) : GenericPtpCamera(session, info, cameraKey) {
    private var remote = false
    private var eventMode = false
    override val capabilities = super.capabilities.copy(
        // R6/R10 card polling is the proven detector. Empty EOS polls previously corrupted the stream.
        eventPolling = !CameraDrivers.cardPollingOnly(info.model) && Ptp.OP_EOS_GET_EVENT in info.operationsSupported,
        physicalShutter = "EOS remote mode with card polling; verify this model and firmware before a shoot.",
    )
    override fun startLive() {
        session.allowUnknownObjectLength()
        session.eosSetRemoteMode(1)
        remote = true
        session.eosSetEventMode(1)
        eventMode = true
    }
    override fun events() = if (capabilities.eventPolling) EosEvents.parse(session.eosGetEvent()).filter { it.isObjectArrival }.map { it.handle } else emptyList()
    override fun keepAlive() { if (Ptp.OP_EOS_KEEP_DEVICE_ON in info.operationsSupported) session.eosKeepDeviceOn() }
    override fun acknowledge(handle: Long) {
        if (Ptp.OP_EOS_TRANSFER_COMPLETE in info.operationsSupported) session.eosTransferComplete(handle)
    }
    override fun close() {
        session.checkCancellation {}
        if (eventMode && session.isUsable) runCatching { session.eosSetEventMode(0) }
        if (remote && session.isUsable) runCatching { session.eosSetRemoteMode(0) }
        session.close()
    }
}

object CameraDrivers {
    internal fun isCanon(vid: Int, info: PtpDeviceInfo) =
        (vid == 0x04a9 || info.manufacturer.equals("Canon", true)) &&
            listOf(Ptp.OP_EOS_SET_REMOTE_MODE, Ptp.OP_EOS_SET_EVENT_MODE).all { it in info.operationsSupported }
    internal fun cardPollingOnly(model: String) = model.uppercase() in setOf("CANON EOS R6", "EOS R6", "CANON EOS R10", "EOS R10")
    internal fun needsWarmUp(vid: Int, pid: Int) = vid == 0x04a9 && pid == 0x32f8

    suspend fun open(manager: UsbManager, device: UsbDevice, live: Boolean = false, genericOnly: Boolean = false): GenericPtpCamera {
        val context = currentCoroutineContext()
        if (live && needsWarmUp(device.vendorId, device.productId)) {
            PtpSession(manager, device).use { warm ->
                warm.checkCancellation { context.ensureActive() }
                checkOpen(warm)
                warm.getStorageIds()
                warm.getObjectHandles(0xffffffffL)
            }
            delay(1000)
        }
        val session = PtpSession(manager, device)
        var driver: GenericPtpCamera? = null
        try {
            session.checkCancellation { context.ensureActive() }
            val info = session.getDeviceInfo()
            checkOpen(session)
            val serial = info.serialNumber.ifBlank { runCatching { device.serialNumber }.getOrNull().orEmpty() }
            // Without a serial, keep recovery scoped to this attachment instead of guessing body identity.
            val key = fingerprint(device.vendorId, device.productId, info.manufacturer, info.model,
                serial.ifBlank { "attachment:${device.deviceName}:${device.deviceId}" })
            driver = if (!genericOnly && isCanon(device.vendorId, info)) CanonEosCamera(session, info, key)
                else GenericPtpCamera(session, info, key)
            if (isCanon(device.vendorId, info)) session.allowUnknownObjectLength()
            if (!driver.capabilities.cardImport) throw PtpException("This USB mode does not advertise card import. Select PTP / Photo Import on the camera.")
            if (live) {
                try { driver.startLive() } catch (e: PtpResponseException) {
                    if (e.code != 0x2005) throw e
                    driver.close()
                    // A clean rejection permits a fresh generic import/poll session.
                    return open(manager, device, live = false, genericOnly = true)
                }
            }
            return driver
        } catch (e: Throwable) {
            if (driver != null) driver.close() else session.close()
            throw e
        }
    }

    private fun checkOpen(session: PtpSession) {
        val code = session.openSession()
        if (code != Ptp.RC_OK && code != Ptp.RC_SESSION_ALREADY_OPEN) throw PtpResponseException(code)
    }
}
