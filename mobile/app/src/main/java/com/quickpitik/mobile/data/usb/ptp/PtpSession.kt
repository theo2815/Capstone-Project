package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Thrown for any PTP transport / protocol failure. */
class PtpException(message: String) : Exception(message)

/**
 * Minimal pure-Kotlin PTP transport over Android USB host. Enough to handshake
 * with a tethered camera (GetDeviceInfo / OpenSession) and report what it
 * supports. The capture loop (Canon EOS_GetEvent polling → GetObject → the
 * existing upload queue) builds on this once the probe confirms the body's
 * capabilities.
 *
 * Not thread-safe — drive it from a single background coroutine. Construction
 * opens + claims the still-image interface and throws if the device can't be
 * opened (e.g. USB permission not granted yet).
 */
class PtpSession(manager: UsbManager, device: UsbDevice) : Closeable {

    private val iface: UsbInterface = findStillImageInterface(device)
        ?: throw PtpException("No USB still-image (PTP) interface on ${device.deviceName}")
    private val bulkIn: UsbEndpoint
    private val bulkOut: UsbEndpoint
    private val interruptIn: UsbEndpoint?
    private val connection: UsbDeviceConnection
    private var transactionId = 0
    private var sessionOpen = false

    val hasInterruptEndpoint: Boolean get() = interruptIn != null

    init {
        var inEp: UsbEndpoint? = null
        var outEp: UsbEndpoint? = null
        var intEp: UsbEndpoint? = null
        for (i in 0 until iface.endpointCount) {
            val ep = iface.getEndpoint(i)
            when (ep.type) {
                UsbConstants.USB_ENDPOINT_XFER_BULK ->
                    if (ep.direction == UsbConstants.USB_DIR_IN) inEp = ep else outEp = ep
                UsbConstants.USB_ENDPOINT_XFER_INT ->
                    if (ep.direction == UsbConstants.USB_DIR_IN) intEp = ep
            }
        }
        bulkIn = inEp ?: throw PtpException("No bulk-IN endpoint")
        bulkOut = outEp ?: throw PtpException("No bulk-OUT endpoint")
        interruptIn = intEp

        connection = manager.openDevice(device)
            ?: throw PtpException("openDevice failed (USB permission not granted?)")
        if (!connection.claimInterface(iface, true)) {
            connection.close()
            throw PtpException("claimInterface failed")
        }
    }

    /** GetDeviceInfo handshake — valid before a session is opened. */
    fun getDeviceInfo(): PtpDeviceInfo {
        sendCommand(Ptp.OP_GET_DEVICE_INFO)
        val data = readContainer()
        val type = containerType(data)
        if (type != Ptp.TYPE_DATA) throw PtpException("GetDeviceInfo: expected data, got type $type")
        checkOk(readContainer(), "GetDeviceInfo")
        return PtpDeviceInfo.parse(data.copyOfRange(CONTAINER_HEADER, data.size))
    }

    /** Open a PTP session; returns the raw response code so the caller can react. */
    fun openSession(sessionId: Int = 1): Int {
        // Per spec the OpenSession transaction carries TransactionID 0; the first
        // operation after it is 1. GetDeviceInfo (outside a session) also used 0.
        transactionId = 0
        sendCommand(Ptp.OP_OPEN_SESSION, sessionId)
        val rc = responseCode(readContainer())
        if (rc == Ptp.RC_OK || rc == Ptp.RC_SESSION_ALREADY_OPEN) sessionOpen = true
        return rc
    }

    /** Close the current PTP session (best-effort — used to clear stale state). */
    fun closeSession() {
        sendCommand(Ptp.OP_CLOSE_SESSION)
        readContainer()
        sessionOpen = false
    }

    // ── Canon EOS tether operations ─────────────────────────────────────────────
    // EOS bodies don't push captures on the interrupt endpoint; after entering
    // remote + event mode the host polls EOS_GetEvent and pulls each new object.

    fun eosSetRemoteMode(mode: Int) = simpleCommand(Ptp.OP_EOS_SET_REMOTE_MODE, mode)
    fun eosSetEventMode(mode: Int) = simpleCommand(Ptp.OP_EOS_SET_EVENT_MODE, mode)
    fun eosKeepDeviceOn() = simpleCommand(Ptp.OP_EOS_KEEP_DEVICE_ON)
    fun eosTransferComplete(handle: Long) = simpleCommand(Ptp.OP_EOS_TRANSFER_COMPLETE, handle.toInt())
    /** Simple one-shot release (no half/full press semantics) — least state. */
    fun eosRemoteRelease() = simpleCommand(Ptp.OP_EOS_REMOTE_RELEASE)
    fun eosRemoteReleaseOn(pressType: Int, extra: Int = 0) =
        simpleCommand(Ptp.OP_EOS_REMOTE_RELEASE_ON, pressType, extra)
    fun eosRemoteReleaseOff(pressType: Int) =
        simpleCommand(Ptp.OP_EOS_REMOTE_RELEASE_OFF, pressType)

    /**
     * Tell the body where a capture goes (EOS SetDevicePropValueEx). Setting the
     * destination to "host" is what makes a host-triggered shot get reported +
     * transferred — without it the camera fires but never notifies us.
     */
    fun eosSetCaptureDestination(dest: Int) {
        // EOS prop-set payload: [UINT32 totalSize][UINT32 propCode][UINT32 value].
        val payload = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
        payload.putInt(12)
        payload.putInt(Ptp.EOS_DPC_CAPTURE_DESTINATION)
        payload.putInt(dest)
        dataOutCommand(Ptp.OP_EOS_SET_DEVICE_PROP_VALUE_EX, payload.array())
    }

    /**
     * Poll the EOS event queue, tolerant of the "no events" case.
     *
     * When its queue is empty the R6 doesn't send a normal data phase — it stalls
     * the bulk-IN endpoint, which reads as -1 and (if unhandled) wedges the pipe so
     * the next command can't even be sent. So: send GetEvent; if no data phase
     * arrives, clear the endpoint halt, drain the response, and return empty bytes
     * (the caller parses zero events and keeps polling). A real event yields a data
     * container we return for parsing.
     */
    fun eosGetEvent(): ByteArray {
        try {
            sendCommand(Ptp.OP_EOS_GET_EVENT)
        } catch (e: Exception) {
            clearHalt(bulkOut)
            sendCommand(Ptp.OP_EOS_GET_EVENT) // one retry after clearing the OUT halt
        }
        val data = readContainerOrNull(GET_EVENT_TIMEOUT_MS)
        if (data == null) {
            clearHalt(bulkIn)
            readContainerOrNull(GET_EVENT_TIMEOUT_MS) // drain the response phase, ignore
            return ByteArray(0)
        }
        if (containerType(data) != Ptp.TYPE_DATA) {
            return ByteArray(0) // got the response directly (no data) = no events
        }
        readContainerOrNull(GET_EVENT_TIMEOUT_MS) // response after data, ignore
        return data.copyOfRange(CONTAINER_HEADER, data.size)
    }

    /** Download a full object (the captured JPEG) by its handle. */
    fun eosGetObject(handle: Long): ByteArray = dataInCommand(Ptp.OP_EOS_GET_OBJECT, handle.toInt())

    // ── Standard PTP card reading (the Zno-style path — primary tether) ─────────
    // After OpenSession, enumerate the memory card and pull objects with the plain
    // standard ops. This is the universal, rock-solid corner of PTP — it works on
    // essentially every camera, unlike the EOS remote-capture extension — and lets
    // us catch each shot the photographer takes with no host trigger and no Live
    // View. (The same ops the OS uses to "import from camera".)

    /** All storage IDs on the camera (usually one or two card slots). */
    fun getStorageIds(): List<Long> =
        parseUint32Array(dataInCommand(Ptp.OP_GET_STORAGE_IDS))

    /**
     * Object handles on a storage. `format` 0 = all formats; `assoc` 0 = all
     * objects regardless of folder (the camera also returns folder associations —
     * the caller filters those out via [getObjectInfo]).
     */
    fun getObjectHandles(storageId: Long, format: Int = 0, assoc: Long = 0): List<Long> =
        parseUint32Array(
            dataInCommand(Ptp.OP_GET_OBJECT_HANDLES, storageId.toInt(), format, assoc.toInt())
        )

    /** Metadata for one object (format + size + filename) — see [PtpObjectInfo]. */
    fun getObjectInfo(handle: Long): PtpObjectInfo =
        PtpObjectInfo.parse(dataInCommand(Ptp.OP_GET_OBJECT_INFO, handle.toInt()))

    /** Download a full object (the JPEG bytes) by handle, standard op. */
    fun getObject(handle: Long): ByteArray = dataInCommand(Ptp.OP_GET_OBJECT, handle.toInt())

    /**
     * Download the camera-rendered thumbnail JPEG for [handle]. Size is camera-
     * defined (typically 160x120 / 240x160) so this comes back in a single
     * data phase and is fast enough to fetch for every visible row during the
     * browse scan. Not every body supports this op — caller should runCatching
     * and fall back to a placeholder.
     */
    fun getThumb(handle: Long): ByteArray = dataInCommand(Ptp.OP_GET_THUMB, handle.toInt())

    /** Parse a PTP AUINT32 array payload: `[UINT32 count][count × UINT32]`. */
    private fun parseUint32Array(data: ByteArray): List<Long> {
        if (data.size < 4) return emptyList()
        val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        val count = bb.int
        val out = ArrayList<Long>(count.coerceIn(0, (data.size - 4) / 4))
        var p = 4
        repeat(count) {
            if (p + 4 > data.size) return@repeat
            out.add(bb.getInt(p).toLong() and 0xFFFFFFFFL)
            p += 4
        }
        return out
    }

    /** Human-readable capability dump driving the Tether Console probe button. */
    fun probe(): String {
        val info = getDeviceInfo()
        val isEos = info.operationsSupported.contains(Ptp.OP_EOS_GET_EVENT)
        val openResult = runCatching { "0x%04X".format(openSession()) }
            .getOrElse { "send failed (${it.message})" }
        return buildString {
            appendLine("${info.manufacturer} ${info.model}".trim())
            appendLine("Serial ${info.serialNumber} • fw ${info.deviceVersion}")
            appendLine("PTP standard 0x%04X".format(info.standardVersion))
            appendLine(
                "Vendor ext: %s (id 0x%X, v0x%04X)".format(
                    Ptp.vendorName(info.vendorExtensionId),
                    info.vendorExtensionId,
                    info.vendorExtensionVersion,
                )
            )
            appendLine("Canon EOS extension: ${if (isEos) "YES — EOS_GetEvent 0x9116 present" else "not detected"}")
            appendLine("Std ObjectAdded event 0x4002: ${if (info.eventsSupported.contains(Ptp.EV_OBJECT_ADDED)) "yes" else "no"}")
            appendLine("Interrupt endpoint: ${if (hasInterruptEndpoint) "present" else "none"}")
            appendLine("OpenSession: $openResult")
            appendLine("Ops (${info.operationsSupported.size}): " + info.operationsSupported.joinToString(" ") { "0x%04X".format(it) })
            appendLine("Events (${info.eventsSupported.size}): " + info.eventsSupported.joinToString(" ") { "0x%04X".format(it) })
        }.trim()
    }

    // ── transport ─────────────────────────────────────────────────────────────

    private fun sendCommand(op: Int, vararg params: Int) {
        val len = CONTAINER_HEADER + params.size * 4
        val buf = ByteBuffer.allocate(len).order(ByteOrder.LITTLE_ENDIAN)
        buf.putInt(len)
        buf.putShort(Ptp.TYPE_COMMAND.toShort())
        buf.putShort(op.toShort())
        buf.putInt(transactionId)
        for (p in params) buf.putInt(p)
        val arr = buf.array()
        val sent = connection.bulkTransfer(bulkOut, arr, arr.size, TIMEOUT_MS)
        transactionId++
        if (sent < 0) throw PtpException("send op 0x%04X failed".format(op))
    }

    /** Send a command that has no data phase; verify the response is OK. */
    private fun simpleCommand(op: Int, vararg params: Int) {
        sendCommand(op, *params)
        checkOk(readContainer(), "op 0x%04X".format(op))
    }

    /** Send a command with a host→device data phase (e.g. SetDevicePropValueEx). */
    private fun dataOutCommand(op: Int, payload: ByteArray, vararg params: Int) {
        val txid = transactionId
        // Command phase.
        val clen = CONTAINER_HEADER + params.size * 4
        val cbuf = ByteBuffer.allocate(clen).order(ByteOrder.LITTLE_ENDIAN)
        cbuf.putInt(clen)
        cbuf.putShort(Ptp.TYPE_COMMAND.toShort())
        cbuf.putShort(op.toShort())
        cbuf.putInt(txid)
        for (p in params) cbuf.putInt(p)
        if (connection.bulkTransfer(bulkOut, cbuf.array(), clen, TIMEOUT_MS) < 0) {
            throw PtpException("op 0x%04X cmd send failed".format(op))
        }
        // Data phase — same TransactionID as the command.
        val dlen = CONTAINER_HEADER + payload.size
        val dbuf = ByteBuffer.allocate(dlen).order(ByteOrder.LITTLE_ENDIAN)
        dbuf.putInt(dlen)
        dbuf.putShort(Ptp.TYPE_DATA.toShort())
        dbuf.putShort(op.toShort())
        dbuf.putInt(txid)
        dbuf.put(payload)
        if (connection.bulkTransfer(bulkOut, dbuf.array(), dlen, TIMEOUT_MS) < 0) {
            throw PtpException("op 0x%04X data send failed".format(op))
        }
        transactionId++
        checkOk(readContainer(), "op 0x%04X".format(op))
    }

    /**
     * Send a command with a device→host data phase and return the data payload
     * (bytes after the 12-byte header). If the device skips the data phase and
     * answers with the response directly, returns an empty array.
     */
    private fun dataInCommand(op: Int, vararg params: Int): ByteArray {
        sendCommand(op, *params)
        val first = readContainer()
        if (containerType(first) != Ptp.TYPE_DATA) {
            val rc = responseCode(first)
            if (rc != Ptp.RC_OK) throw PtpException("op 0x%04X resp 0x%04X (no data)".format(op, rc))
            return ByteArray(0)
        }
        checkOk(readContainer(), "op 0x%04X".format(op))
        return first.copyOfRange(CONTAINER_HEADER, first.size)
    }

    /**
     * Read one full PTP container. PTP ends a data phase either by reaching the
     * length declared in the 12-byte header OR via a short/zero USB packet. We
     * honour both: small responses (DeviceInfo, GetEvent) finish on the first
     * short read; a full JPEG via GetObject loops until the declared length;
     * Canon's occasional 0xFFFFFFFF "unknown length" phase loops until a short
     * packet arrives.
     */
    private fun readContainer(timeoutMs: Int = TIMEOUT_MS): ByteArray {
        val first = ByteArray(READ_CHUNK)
        val n = connection.bulkTransfer(bulkIn, first, first.size, timeoutMs)
        if (n < 4) throw PtpException("short/failed read ($n bytes)")
        val declared = ByteBuffer.wrap(first, 0, 4).order(ByteOrder.LITTLE_ENDIAN).int
        val lengthKnown = declared in CONTAINER_HEADER..MAX_CONTAINER
        val out = ByteArrayOutputStream(if (lengthKnown) declared else maxOf(n, READ_CHUNK))
        out.write(first, 0, n)
        // A first read shorter than our buffer means a short packet ended the
        // transfer — done — unless the declared length says more is still coming.
        if (n < first.size && !(lengthKnown && out.size() < declared)) {
            return out.toByteArray()
        }
        while (true) {
            if (lengthKnown && out.size() >= declared) break
            val more = ByteArray(READ_CHUNK)
            val m = connection.bulkTransfer(bulkIn, more, more.size, timeoutMs)
            if (m <= 0) break
            out.write(more, 0, m)
            if (!lengthKnown && m < more.size) break // short packet ends the phase
        }
        return out.toByteArray()
    }

    /** [readContainer] that yields null instead of throwing (for tolerant polls). */
    private fun readContainerOrNull(timeoutMs: Int = TIMEOUT_MS): ByteArray? =
        runCatching { readContainer(timeoutMs) }.getOrNull()

    /**
     * Clear a stalled (halted) endpoint via CLEAR_FEATURE(ENDPOINT_HALT):
     * bmRequestType 0x02 (host→device, standard, recipient = endpoint),
     * bRequest 1, wValue 0, wIndex = endpoint address.
     */
    private fun clearHalt(ep: UsbEndpoint) {
        runCatching { connection.controlTransfer(0x02, 1, 0, ep.address, null, 0, 1000) }
    }

    private fun containerType(c: ByteArray): Int =
        ByteBuffer.wrap(c, 4, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF

    private fun responseCode(c: ByteArray): Int =
        ByteBuffer.wrap(c, 6, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF

    private fun checkOk(resp: ByteArray, what: String) {
        val rc = responseCode(resp)
        if (rc != Ptp.RC_OK) throw PtpException("$what response 0x%04X".format(rc))
    }

    override fun close() {
        runCatching { if (sessionOpen) sendCommand(Ptp.OP_CLOSE_SESSION) }
        runCatching { readContainer() } // drain the CloseSession response, best-effort
        runCatching { connection.releaseInterface(iface) }
        runCatching { connection.close() }
    }

    private companion object {
        const val CONTAINER_HEADER = 12
        const val READ_CHUNK = 16384
        const val TIMEOUT_MS = 5000
        const val GET_EVENT_TIMEOUT_MS = 1500 // short — empty polls shouldn't hang
        const val MAX_CONTAINER = 256 * 1024 * 1024 // sanity ceiling for one data phase

        fun findStillImageInterface(device: UsbDevice): UsbInterface? {
            for (i in 0 until device.interfaceCount) {
                val iface = device.getInterface(i)
                if (iface.interfaceClass == UsbConstants.USB_CLASS_STILL_IMAGE) return iface
            }
            return null
        }
    }
}
