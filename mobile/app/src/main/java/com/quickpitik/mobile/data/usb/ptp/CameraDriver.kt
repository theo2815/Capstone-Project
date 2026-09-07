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
    val transientObjects: Boolean = false,
)

/** A mode/format choice needs user action, not five minutes of automatic reconnects. */
class CameraSetupException(message: String) : PtpException(message)

/** Protocol behavior only. Account/event persistence belongs above this boundary. */
open class GenericPtpCamera(
    val session: PtpSession,
    val info: PtpDeviceInfo,
    val cameraKey: String,
) : Closeable {
    open val capabilities = CameraCapabilities(
        cardImport = listOf(Ptp.OP_GET_OBJECT_HANDLES, Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT)
            .all { it in info.operationsSupported },
        thumbnails = Ptp.OP_GET_THUMB in info.operationsSupported,
        eventPolling = session.hasInterruptEndpoint,
        physicalShutter = "Unverified; use card import if this USB mode locks the shutter.",
        imageFormats = info.imageFormats,
    )
    open fun startLive() {}
    open fun events(): List<Long> = session.interruptEvents().filter {
        it.code == Ptp.EV_OBJECT_ADDED ||
            (info.manufacturer.startsWith("Panasonic", true) && it.code == Ptp.EV_PANASONIC_OBJECT_ADDED)
    }.mapNotNull { it.handle }
    open fun nextCapture(): Pair<Long, PtpObjectInfo>? = null
    open fun keepAlive() {}
    open fun acknowledge(handle: Long) {}
    open fun handles(): List<Long> {
        if (Ptp.OP_GET_STORAGE_IDS !in info.operationsSupported) return session.getObjectHandles(0xffffffffL)
        val storages = session.getStorageIds()
        if (storages.isEmpty()) return session.getObjectHandles(0xffffffffL)
        val handles = LinkedHashSet<Long>()
        var readable = false
        var unavailable: PtpResponseException? = null
        for (storage in storages) {
            try {
                handles.addAll(session.getObjectHandles(storage))
                readable = true
            } catch (e: PtpResponseException) {
                if (e.code != Ptp.RC_STORE_NOT_AVAILABLE && e.code != Ptp.RC_INVALID_STORAGE_ID) throw e
                unavailable = e
            }
        }
        // Never turn an entirely failed scan into a successful empty baseline.
        if (!readable) throw requireNotNull(unavailable)
        return handles.toList()
    }
    open fun objectInfo(handle: Long) = session.getObjectInfo(handle)
    open fun readObject(handle: Long, sink: OutputStream, expectedSize: Long?) = session.getObjectTo(handle, sink, expectedSize)
    override fun close() = session.close()
}

class CanonEosCamera(session: PtpSession, info: PtpDeviceInfo, cameraKey: String) : GenericPtpCamera(session, info, cameraKey) {
    private var remote = false
    private var eventMode = false
    override val capabilities = super.capabilities.copy(
        // R6/R10 card polling is the proven detector. Empty EOS polls previously corrupted the stream.
        eventPolling = super.capabilities.eventPolling ||
            (!CameraDrivers.cardPollingOnly(info.model) && Ptp.OP_EOS_GET_EVENT in info.operationsSupported),
        physicalShutter = "EOS remote mode with card polling; verify this model and firmware before a shoot.",
    )
    override fun startLive() {
        session.allowUnknownObjectLength()
        session.eosSetRemoteMode(1)
        remote = true
        session.eosSetEventMode(1)
        eventMode = true
    }
    override fun events(): List<Long> = super.events() +
        if (!CameraDrivers.cardPollingOnly(info.model) && Ptp.OP_EOS_GET_EVENT in info.operationsSupported)
            EosEvents.parse(session.eosGetEvent()).filter { it.isObjectArrival }.map { it.handle }
        else emptyList()
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

class NikonPtpCamera(session: PtpSession, info: PtpDeviceInfo, cameraKey: String) : GenericPtpCamera(session, info, cameraKey) {
    private val eventOps = if (CameraDrivers.nikonEventQuirk(info.model)) mutableListOf() else
        listOf(Ptp.OP_NIKON_GET_EVENT_EX, Ptp.OP_NIKON_GET_EVENT).filter { it in info.operationsSupported }.toMutableList()
    override val capabilities get() = super.capabilities.copy(
        eventPolling = super.capabilities.eventPolling || eventOps.isNotEmpty(),
        physicalShutter = "Nikon card monitoring; keep camera control enabled and verify shooting in MTP/PTP mode.",
    )
    override fun events(): List<Long> {
        val handles = super.events().toMutableList()
        while (eventOps.isNotEmpty()) {
            val op = eventOps.first()
            try {
                handles += PtpEvent.parseNikon(session.vendorData(op), op == Ptp.OP_NIKON_GET_EVENT_EX)
                    .filter { it.code == Ptp.EV_OBJECT_ADDED }.mapNotNull { it.handle }
                break
            } catch (e: PtpResponseException) {
                if (e.code != Ptp.RC_OPERATION_NOT_SUPPORTED) throw e
                eventOps.removeAt(0)
            }
        }
        return handles.distinct()
    }
}

object CameraDrivers {
    internal fun isCanon(vid: Int, info: PtpDeviceInfo) =
        (vid == 0x04a9 || info.manufacturer.startsWith("Canon", true)) &&
            listOf(Ptp.OP_EOS_SET_REMOTE_MODE, Ptp.OP_EOS_SET_EVENT_MODE).all { it in info.operationsSupported }
    internal fun cardPollingOnly(model: String) = model.uppercase() in setOf("CANON EOS R6", "EOS R6", "CANON EOS R10", "EOS R10")
    internal fun needsWarmUp(vid: Int, pid: Int) = vid == 0x04a9 && pid == 0x32f8
    internal fun nikonEventQuirk(model: String) = Regex("(?i)\\bNikon\\s+1\\b").containsMatchIn(model)
    internal fun isSony(vid: Int, info: PtpDeviceInfo) =
        (vid == 0x054c || info.manufacturer.startsWith("Sony", true)) &&
            listOf(0x9201, 0x9202).all { it in info.operationsSupported }

    fun usbGuidance(vid: Int): String = when (vid) {
        0x04a9 -> "Canon: start auto-upload to enter EOS mode. If the shutter stays locked, unplug to shoot, then use card import."
        0x04b0 -> "Nikon: choose MTP/PTP and keep camera control enabled. If this mode locks shooting, use card import after shooting."
        0x054c -> "Sony: use PC Remote, save to PC + Camera, keep Camera priority, and send JPEGs to the PC. Use MTP for card import."
        0x04cb -> "Fujifilm: use USB CARD READER for import. USB TETHER support varies; confirm photos also save to the camera card."
        0x04da -> "Panasonic: use PTP/MTP for import or a supported PC(Tether) mode for shooting. Keep saving to the camera card enabled."
        else -> "Choose PTP/MTP for card import. Live shooting depends on the camera's USB mode; confirm a test photo saves to the card."
    }

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
            val info = session.identify()
            val serial = info.serialNumber.ifBlank { runCatching { device.serialNumber }.getOrNull().orEmpty() }
            // Without a serial, keep recovery scoped to this attachment instead of guessing body identity.
            val key = fingerprint(device.vendorId, device.productId, info.manufacturer, info.model,
                serial.ifBlank { "attachment:${device.deviceName}:${device.deviceId}" })
            driver = when {
                genericOnly -> GenericPtpCamera(session, info, key)
                isCanon(device.vendorId, info) -> CanonEosCamera(session, info, key)
                live && isSony(device.vendorId, info) -> SonyPtpCamera(session, info, key)
                device.vendorId == 0x04b0 || info.manufacturer.startsWith("Nikon", true) -> NikonPtpCamera(session, info, key)
                else -> GenericPtpCamera(session, info, key)
            }
            if (isCanon(device.vendorId, info)) session.allowUnknownObjectLength()
            if (live) {
                try { driver.startLive() } catch (e: PtpResponseException) {
                    if (driver is SonyPtpCamera && e.code == Ptp.RC_OPERATION_NOT_SUPPORTED)
                        throw CameraSetupException("This Sony USB mode rejects PC Remote setup. Select PC Remote or use MTP card import.")
                    if (driver !is CanonEosCamera || e.code != Ptp.RC_OPERATION_NOT_SUPPORTED) throw e
                    driver.close()
                    // A clean rejection permits a fresh generic import/poll session.
                    return open(manager, device, live = false, genericOnly = true)
                }
            }
            if (!driver.capabilities.cardImport && !(live && driver.capabilities.transientObjects))
                throw CameraSetupException("This USB mode does not expose readable photos. ${usbGuidance(device.vendorId)}")
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
