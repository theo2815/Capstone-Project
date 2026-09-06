package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import java.io.Closeable
import java.io.IOException
import java.util.concurrent.Semaphore

open class PtpException(message: String, cause: Throwable? = null) : IOException(message, cause)
class PtpResponseException(val code: Int) : PtpException("Camera response 0x%04X".format(code))

/** One USB transfer, not necessarily one PTP container. Also used by scripted tests. */
internal interface PtpTransport : Closeable {
    val hasInterruptEndpoint: Boolean get() = false
    fun read(buffer: ByteArray, timeoutMs: Int): Int
    fun write(buffer: ByteArray, timeoutMs: Int): Int
}

internal class AndroidPtpTransport(manager: UsbManager, device: UsbDevice) : PtpTransport {
    private val iface = (0 until device.interfaceCount).map(device::getInterface)
        .firstOrNull { it.interfaceClass == UsbConstants.USB_CLASS_STILL_IMAGE }
        ?: throw PtpException("Select the camera's PTP / PC connection USB mode.")
    private val endpoints = (0 until iface.endpointCount).map(iface::getEndpoint)
    private val bulkIn = endpoints.firstOrNull {
        it.type == UsbConstants.USB_ENDPOINT_XFER_BULK && it.direction == UsbConstants.USB_DIR_IN
    } ?: throw PtpException("Camera has no bulk-IN endpoint.")
    private val bulkOut = endpoints.firstOrNull {
        it.type == UsbConstants.USB_ENDPOINT_XFER_BULK && it.direction == UsbConstants.USB_DIR_OUT
    } ?: throw PtpException("Camera has no bulk-OUT endpoint.")
    override val hasInterruptEndpoint = endpoints.any {
        it.type == UsbConstants.USB_ENDPOINT_XFER_INT && it.direction == UsbConstants.USB_DIR_IN
    }
    private val connection = run {
        if (!ownership.tryAcquire()) throw PtpException("Camera is finishing another transfer. Try again shortly.")
        try {
            val opened = manager.openDevice(device) ?: throw PtpException("USB permission is required.")
            try {
                if (!opened.claimInterface(iface, true)) throw PtpException("Camera interface is busy.")
                opened
            } catch (e: Throwable) {
                opened.close()
                throw e
            }
        } catch (e: Throwable) {
            ownership.release()
            throw e
        }
    }
    private var closed = false

    override fun read(buffer: ByteArray, timeoutMs: Int) =
        connection.bulkTransfer(bulkIn, buffer, buffer.size, timeoutMs)

    override fun write(buffer: ByteArray, timeoutMs: Int) =
        connection.bulkTransfer(bulkOut, buffer, buffer.size, timeoutMs)

    override fun close() {
        if (closed) return
        closed = true
        try {
            try { connection.releaseInterface(iface) } finally { connection.close() }
        } finally { ownership.release() }
    }

    private companion object {
        // ponytail: one camera per phone; use per-device leases if multi-camera is introduced.
        val ownership = Semaphore(1, true)
    }
}
