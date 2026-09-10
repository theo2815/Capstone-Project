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
    /** Behaviour the driver relies on without a hardware pass on this body; logged, never silent. */
    val assumptions: List<String> = emptyList(),
)

/** A mode/format choice needs user action, not five minutes of automatic reconnects. */
class CameraSetupException(message: String) : PtpException(message)

/** Protocol behavior only. Account/event persistence belongs above this boundary. */
open class GenericPtpCamera(
    val session: PtpSession,
    info: PtpDeviceInfo,
    val cameraKey: String,
    val quirks: CameraQuirks = CameraQuirks.forCamera(0, 0, info.model),
) : Closeable {
    /** Re-read after vendor setup: the advertised operation set can change (libgphoto2: EOS M2). */
    var info: PtpDeviceInfo = info
        protected set
    /** Set when vendor setup was declined and the session carried on as standard PTP. */
    var setupWarning: String? = null
    open val capabilities: CameraCapabilities get() = CameraCapabilities(
        cardImport = listOf(Ptp.OP_GET_OBJECT_HANDLES, Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT)
            .all { it in info.operationsSupported },
        thumbnails = Ptp.OP_GET_THUMB in info.operationsSupported,
        eventPolling = session.hasInterruptEndpoint,
        physicalShutter = "Unverified; use card import if this USB mode locks the shutter.",
        imageFormats = info.imageFormats,
        assumptions = listOf("Standard PTP: the camera USB mode decides whether the shutter stays usable; new shots are found by card scans plus ObjectAdded events."),
    )
    /** Vendor PC-link setup every session needs (browse, import and live alike); nothing on generic PTP. */
    open fun startSession() {}
    open fun startLive() {}
    /** True when live capture without the vendor setup is known to lock the shutter or lose frames. */
    open val liveNeedsVendorSetup: Boolean get() = false
    open val vendorRejectedMessage: String get() = "The camera rejected its USB setup. Check the camera USB mode and start again."
    /** Vendor codes that also mean a new object exists; the standard 0x4002 is always accepted. */
    protected open val objectAddedEvents: Set<Int> get() = emptySet()
    open fun events(): List<Long> = session.interruptEvents()
        .filter { it.code == Ptp.EV_OBJECT_ADDED || it.code in objectAddedEvents }.mapNotNull { it.handle }
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
    protected fun refreshInfo() { info = session.getDeviceInfo() }
    override fun close() = session.close()
}

class CanonEosCamera(
    session: PtpSession, info: PtpDeviceInfo, cameraKey: String,
    quirks: CameraQuirks = CameraQuirks.forCamera(0, 0, info.model),
) : GenericPtpCamera(session, info, cameraKey, quirks) {
    private var remote = false
    private var eventMode = false
    override val liveNeedsVendorSetup get() = true
    override val vendorRejectedMessage get() = "Canon rejected live USB setup. Check the camera USB mode, or stop and use card import after shooting."
    override val capabilities get() = super.capabilities.copy(
        eventPolling = session.hasInterruptEndpoint || (quirks.vendorEvents && Ptp.OP_EOS_GET_EVENT in info.operationsSupported),
        physicalShutter = "EOS remote mode with card polling; verify this model and firmware before a shoot.",
        assumptions = listOfNotNull(
            "Remote mode 0x%X keeps the display and shutter live and is left on at close (verified on EOS R6 and R10 only).".format(quirks.eosRemoteMode),
            quirks.note,
        ),
    )
    // libgphoto2 (camlibs/ptp2/library.c camera_init) enters remote mode for every EOS it
    // opens, even for plain file listing: a bare session parks the body on its PC-connection
    // screen with the shutter dead (R6, 2026-05-26); remote mode keeps it live (R6/R10, 2026-09-02/04).
    override fun startSession() {
        session.eosSetRemoteMode(quirks.eosRemoteMode)
        remote = true
        refreshInfo()
    }
    override fun startLive() {
        session.eosSetEventMode(1)
        eventMode = true
    }
    override fun events(): List<Long> = super.events() +
        if (quirks.vendorEvents && Ptp.OP_EOS_GET_EVENT in info.operationsSupported)
            EosEvents.parse(session.eosGetEvent()).filter { it.isObjectArrival }.map { it.handle }
        else emptyList()
    override fun keepAlive() { if (Ptp.OP_EOS_KEEP_DEVICE_ON in info.operationsSupported) session.eosKeepDeviceOn() }
    override fun acknowledge(handle: Long) {
        if (Ptp.OP_EOS_TRANSFER_COMPLETE in info.operationsSupported) session.eosTransferComplete(handle)
    }
    override fun close() {
        session.checkCancellation {}
        if (eventMode && session.isUsable) runCatching { session.eosSetEventMode(0) }
        if (remote && session.isUsable) runCatching {
            // libgphoto2 config.c camera_unprepare_canon_eos_capture: mode 0 "resets the camera
            // state", mode 1 "enables camera display". Ending on 0 leaves the body locked on its
            // PC-connection screen until it is unplugged; ending on 1 hands it back to the photographer.
            session.eosSetRemoteMode(0)
            session.eosSetRemoteMode(1)
        }
        session.close()
    }
}

class NikonPtpCamera(
    session: PtpSession, info: PtpDeviceInfo, cameraKey: String,
    quirks: CameraQuirks = CameraQuirks.forCamera(0, 0, info.model),
) : GenericPtpCamera(session, info, cameraKey, quirks) {
    private val eventOps = if (!quirks.vendorEvents) mutableListOf() else
        listOf(Ptp.OP_NIKON_GET_EVENT_EX, Ptp.OP_NIKON_GET_EVENT).filter { it in info.operationsSupported }.toMutableList()
    override val capabilities get() = super.capabilities.copy(
        eventPolling = super.capabilities.eventPolling || eventOps.isNotEmpty(),
        physicalShutter = "Nikon card monitoring; keep camera control enabled and verify shooting in MTP/PTP mode.",
        assumptions = listOfNotNull(
            "A plain MTP/PTP session leaves the body shootable (gphoto2 tethered capture relies on this on Nikon DSLRs); Z bodies are unverified here. No control-mode or ApplicationMode writes are sent.",
            quirks.note,
        ),
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

/** Fujifilm X/GFX: tether modes push new shots as objects the card scan picks up; card-reader mode is import-only. */
class FujiPtpCamera(
    session: PtpSession, info: PtpDeviceInfo, cameraKey: String,
    quirks: CameraQuirks = CameraQuirks.forCamera(0, 0, info.model),
) : GenericPtpCamera(session, info, cameraKey, quirks) {
    override val objectAddedEvents get() = setOf(Ptp.EV_FUJI_OBJECT_ADDED)
    override val capabilities get() = super.capabilities.copy(
        physicalShutter = "Fujifilm: USB TETHER SHOOTING AUTO keeps the shutter on the body and saves to card; USB CARD READER is import-only.",
        assumptions = listOfNotNull(
            "New tether-mode shots appear in GetObjectHandles and are found by the card scan (libgphoto2 polls the same way).",
            "Priority Mode is read and, only if it reports USB, set back to Camera so the body keeps its controls (the libgphoto2 fix for a locked Fujifilm shutter).",
            quirks.note,
        ),
    )
    override fun startSession() {
        if (Ptp.FUJI_DPC_PRIORITY_MODE !in info.devicePropertiesSupported) return
        if (session.getDevicePropValueU16(Ptp.FUJI_DPC_PRIORITY_MODE) == Ptp.FUJI_PRIORITY_USB)
            session.setDevicePropValueU16(Ptp.FUJI_DPC_PRIORITY_MODE, Ptp.FUJI_PRIORITY_CAMERA)
    }
}

/** LUMIX PC(Tether): standard PTP plus LUMIX object-added events; host-memory transfers are not implemented. */
class PanasonicPtpCamera(
    session: PtpSession, info: PtpDeviceInfo, cameraKey: String,
    quirks: CameraQuirks = CameraQuirks.forCamera(0, 0, info.model),
) : GenericPtpCamera(session, info, cameraKey, quirks) {
    override val objectAddedEvents get() = setOf(Ptp.EV_PANASONIC_OBJECT_ADDED, Ptp.EV_PANASONIC_OBJECT_ADDED_SDRAM)
    override val capabilities get() = super.capabilities.copy(
        physicalShutter = "Panasonic PC(Tether): card scan plus LUMIX object events; body-shutter delivery is unverified.",
        assumptions = listOfNotNull(
            "PC(Tether) leaves the body shootable and lists new shots on the card; PC(Storage) is mass storage and cannot be read here.",
            quirks.note,
        ),
    )
}

object CameraDrivers {
    internal fun isCanon(vid: Int, info: PtpDeviceInfo) =
        (vid == 0x04a9 || info.manufacturer.startsWith("Canon", true)) &&
            listOf(Ptp.OP_EOS_SET_REMOTE_MODE, Ptp.OP_EOS_SET_EVENT_MODE).all { it in info.operationsSupported }
    internal fun cardPollingOnly(model: String) = !CameraQuirks.forCamera(0, 0, model).vendorEvents
    internal fun needsWarmUp(vid: Int, pid: Int) = CameraQuirks.forCamera(vid, pid, "").warmUp
    internal fun nikonEventQuirk(model: String) = !CameraQuirks.forCamera(0, 0, model).vendorEvents
    internal fun isSony(vid: Int, info: PtpDeviceInfo) =
        (vid == 0x054c || info.manufacturer.startsWith("Sony", true)) &&
            listOf(0x9201, 0x9202).all { it in info.operationsSupported }
    private fun isVendor(vid: Int, info: PtpDeviceInfo, id: Int, prefix: String) =
        vid == id || info.manufacturer.startsWith(prefix, true)

    fun usbGuidance(vid: Int): String = when (vid) {
        0x04a9 -> "Canon: start auto-upload to enter EOS mode. If the shutter stays locked, unplug to shoot, then use card import."
        0x04b0 -> "Nikon: choose MTP/PTP and keep camera control enabled. If this mode locks shooting, use card import after shooting."
        0x054c -> "Sony: use PC Remote, save to PC + Camera, keep Camera priority, and send JPEGs to the PC. Use MTP for card import."
        0x04cb -> "Fujifilm: use USB TETHER SHOOTING AUTO to shoot with the body and save to card; USB CARD READER is for import only."
        0x04da -> "Panasonic: use PC(Tether) for shooting or PTP/MTP for import. PC(Storage) cannot be read here. Keep saving to the camera card enabled."
        0x07b4 -> "Olympus / OM System: choose the MTP USB mode for card import; Control mode is for OM Capture only."
        0x0a17, 0x25fb -> "Pentax / Ricoh: choose the MTP/PTP USB mode for card import."
        0x1a98 -> "Leica: choose the PTP USB mode for card import."
        0x1003 -> "Sigma: choose the PTP/MTP USB mode for card import."
        else -> "Choose PTP/MTP for card import. Live shooting depends on the camera USB mode; confirm a test photo saves to the card."
    }

    suspend fun open(manager: UsbManager, device: UsbDevice, live: Boolean = false, warmUp: Boolean = true): GenericPtpCamera {
        val context = currentCoroutineContext()
        // Every Canon session sends a vendor op, and the R10 stalls on its first one after attach.
        if (warmUp && CameraQuirks.forCamera(device.vendorId, device.productId, "").warmUp) {
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
            val quirks = CameraQuirks.forCamera(device.vendorId, device.productId, info.model)
            val serial = info.serialNumber.ifBlank { runCatching { device.serialNumber }.getOrNull().orEmpty() }
            // Without a serial, keep recovery scoped to this attachment instead of guessing body identity.
            val key = fingerprint(device.vendorId, device.productId, info.manufacturer, info.model,
                serial.ifBlank { "attachment:${device.deviceName}:${device.deviceId}" })
            driver = when {
                isCanon(device.vendorId, info) -> CanonEosCamera(session, info, key, quirks)
                live && isSony(device.vendorId, info) -> SonyPtpCamera(session, info, key, quirks)
                isVendor(device.vendorId, info, 0x04b0, "Nikon") -> NikonPtpCamera(session, info, key, quirks)
                isVendor(device.vendorId, info, 0x04cb, "Fuji") -> FujiPtpCamera(session, info, key, quirks)
                isVendor(device.vendorId, info, 0x04da, "Panasonic") -> PanasonicPtpCamera(session, info, key, quirks)
                else -> GenericPtpCamera(session, info, key, quirks)
            }
            // A 0xFFFFFFFF data length is legal PTP for any vendor; object reads stay bounded by ObjectInfo size and MAX_OBJECT.
            session.allowUnknownObjectLength()
            try {
                driver.startSession()
                if (live) driver.startLive()
            } catch (e: PtpResponseException) {
                when {
                    // Setup that shooting does not depend on degrades to standard PTP instead of failing the session.
                    !(live && driver.liveNeedsVendorSetup) ->
                        driver.setupWarning = "Camera declined vendor setup (0x%04X); continuing with standard PTP.".format(e.code)
                    // Generic import mode can lock the shutter. Never silently switch a live shoot to it.
                    e.code == Ptp.RC_OPERATION_NOT_SUPPORTED -> throw CameraSetupException(driver.vendorRejectedMessage)
                    else -> throw e
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
