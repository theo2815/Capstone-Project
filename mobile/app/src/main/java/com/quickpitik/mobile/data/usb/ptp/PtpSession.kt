package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import java.io.Closeable
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** One exclusively owned PTP session. All callers use the same checked framing. */
class PtpSession internal constructor(private val transport: PtpTransport) : Closeable {
    constructor(manager: UsbManager, device: UsbDevice) : this(AndroidPtpTransport(manager, device))
    private val wire = PtpWire(transport)
    private var sessionOpen = false
    private var closed = false
    val hasInterruptEndpoint get() = transport.hasInterruptEndpoint
    val isUsable get() = wire.usable && !closed
    fun checkCancellation(check: () -> Unit) { wire.checkCancelled = check }
    fun allowUnknownObjectLength() { wire.allowUnknownObjectLength = true }

    fun getDeviceInfo(): PtpDeviceInfo = PtpDeviceInfo.parse(wire.data(Ptp.OP_GET_DEVICE_INFO))

    fun openSession(sessionId: Int = 1): Int {
        wire.transactionId = 0
        val code = wire.response(Ptp.OP_OPEN_SESSION, sessionId)
        sessionOpen = code == Ptp.RC_OK || code == Ptp.RC_SESSION_ALREADY_OPEN
        return code
    }

    fun closeSession() {
        PtpWire.requireOk(wire.response(Ptp.OP_CLOSE_SESSION))
        sessionOpen = false
    }

    fun eosSetRemoteMode(mode: Int) = simple(Ptp.OP_EOS_SET_REMOTE_MODE, mode)
    fun eosSetEventMode(mode: Int) = simple(Ptp.OP_EOS_SET_EVENT_MODE, mode)
    fun eosKeepDeviceOn() = simple(Ptp.OP_EOS_KEEP_DEVICE_ON)
    fun eosTransferComplete(handle: Long) = simple(Ptp.OP_EOS_TRANSFER_COMPLETE, handle.toInt())
    fun eosRemoteRelease() = simple(Ptp.OP_EOS_REMOTE_RELEASE)
    fun eosRemoteReleaseOn(pressType: Int, extra: Int = 0) = simple(Ptp.OP_EOS_REMOTE_RELEASE_ON, pressType, extra)
    fun eosRemoteReleaseOff(pressType: Int) = simple(Ptp.OP_EOS_REMOTE_RELEASE_OFF, pressType)
    fun eosSetCaptureDestination(dest: Int) = wire.sendData(
        Ptp.OP_EOS_SET_DEVICE_PROP_VALUE_EX,
        ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
            .putInt(12).putInt(Ptp.EOS_DPC_CAPTURE_DESTINATION).putInt(dest).array(),
    )
    fun eosGetEvent(): ByteArray = wire.data(Ptp.OP_EOS_GET_EVENT)
    fun eosGetObject(handle: Long): ByteArray = wire.data(Ptp.OP_EOS_GET_OBJECT, handle.toInt())
    fun getStorageIds(): List<Long> = parseUint32Array(wire.data(Ptp.OP_GET_STORAGE_IDS))
    fun getObjectHandles(storageId: Long, format: Int = 0, assoc: Long = 0): List<Long> =
        parseUint32Array(wire.data(Ptp.OP_GET_OBJECT_HANDLES, storageId.toInt(), format, assoc.toInt()))
    fun getObjectInfo(handle: Long): PtpObjectInfo =
        PtpObjectInfo.parse(wire.data(Ptp.OP_GET_OBJECT_INFO, handle.toInt()))
    fun getObject(handle: Long): ByteArray = wire.data(Ptp.OP_GET_OBJECT, handle.toInt())
    fun getObjectTo(handle: Long, sink: OutputStream, expectedSize: Long? = null): Long =
        wire.objectTo(Ptp.OP_GET_OBJECT, handle, sink, expectedSize)
    fun eosGetObjectTo(handle: Long, sink: OutputStream, expectedSize: Long? = null): Long =
        wire.objectTo(Ptp.OP_EOS_GET_OBJECT, handle, sink, expectedSize)
    fun getThumb(handle: Long): ByteArray = wire.data(Ptp.OP_GET_THUMB, handle.toInt())
    private fun simple(op: Int, vararg params: Int) = PtpWire.requireOk(wire.response(op, *params))

    private fun parseUint32Array(data: ByteArray): List<Long> {
        if (data.size < 4) throw PtpException("Truncated PTP array.")
        val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        val count = bb.int
        if (count < 0 || count > 100_000 || count > bb.remaining() / 4) throw PtpException("Invalid PTP array count.")
        return List(count) { bb.int.toLong() and 0xffffffffL }
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


    /** Legacy probes must reopen instead of issuing speculative USB reset requests. */
    fun resetDevice() = close()

    override fun close() {
        if (closed) return
        wire.checkCancelled = {}
        try {
            if (sessionOpen && wire.usable) runCatching { closeSession() }
        } finally {
            closed = true
            transport.close()
        }
    }
}
