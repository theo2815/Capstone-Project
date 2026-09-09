package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import android.hardware.usb.UsbRequest
import java.io.Closeable
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeoutException

open class PtpException(message: String, cause: Throwable? = null) : IOException(message, cause)
class PtpResponseException(val code: Int) : PtpException("Camera response 0x%04X".format(code))

/** One USB transfer, not necessarily one PTP container. Also used by scripted tests. */
internal interface PtpTransport : Closeable {
    val hasInterruptEndpoint: Boolean get() = false
    fun read(buffer: ByteArray, timeoutMs: Int): Int
    fun write(buffer: ByteArray, timeoutMs: Int): Int
    fun readEvent(timeoutMs: Int): ByteArray? = null
}

/** Discovery and opening must agree on an interface that can actually carry PTP. */
internal object UsbPtpInterfaces {
    fun find(device: UsbDevice): UsbInterface? = find((0 until device.interfaceCount).map(device::getInterface))
    fun find(interfaces: List<UsbInterface>): UsbInterface? = interfaces.filter { iface ->
        iface.interfaceClass == UsbConstants.USB_CLASS_STILL_IMAGE &&
            listOf(UsbConstants.USB_DIR_IN, UsbConstants.USB_DIR_OUT).all { direction ->
                (0 until iface.endpointCount).any { i -> iface.getEndpoint(i).let {
                    it.type == UsbConstants.USB_ENDPOINT_XFER_BULK && it.direction == direction
                } }
            }
    }.minByOrNull { it.alternateSetting }
}

internal class AndroidPtpTransport(manager: UsbManager, device: UsbDevice) : PtpTransport {
    private val iface = UsbPtpInterfaces.find(device)
        ?: throw PtpException("Select the camera's PTP / PC connection USB mode.")
    private val endpoints = (0 until iface.endpointCount).map(iface::getEndpoint)
    private val bulkIn = endpoints.firstOrNull {
        it.type == UsbConstants.USB_ENDPOINT_XFER_BULK && it.direction == UsbConstants.USB_DIR_IN
    } ?: throw PtpException("Camera has no bulk-IN endpoint.")
    private val bulkOut = endpoints.firstOrNull {
        it.type == UsbConstants.USB_ENDPOINT_XFER_BULK && it.direction == UsbConstants.USB_DIR_OUT
    } ?: throw PtpException("Camera has no bulk-OUT endpoint.")
    private val interrupt = endpoints.firstOrNull {
        it.type == UsbConstants.USB_ENDPOINT_XFER_INT && it.direction == UsbConstants.USB_DIR_IN
    }
    override val hasInterruptEndpoint get() = interrupt != null
    private val connection = run {
        if (!ownership.tryAcquire()) throw PtpException("Camera is finishing another transfer. Try again shortly.")
        try {
            val opened = manager.openDevice(device) ?: throw PtpException("USB permission is required.")
            try {
                if (!opened.claimInterface(iface, true)) throw PtpException("Camera interface is busy.")
                if (iface.alternateSetting != 0 && !opened.setInterface(iface))
                    throw PtpException("Camera USB alternate interface could not be selected.")
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
    private var eventRequest: UsbRequest? = null
    private val eventBuffer = ByteBuffer.allocate(512)
    private var eventQueued = false

    override fun read(buffer: ByteArray, timeoutMs: Int) =
        connection.bulkTransfer(bulkIn, buffer, buffer.size, timeoutMs)

    override fun write(buffer: ByteArray, timeoutMs: Int) =
        connection.bulkTransfer(bulkOut, buffer, buffer.size, timeoutMs)

    override fun readEvent(timeoutMs: Int): ByteArray? {
        val endpoint = interrupt ?: return null
        if (closed) throw PtpException("Camera USB connection is closed.")
        val request = eventRequest ?: UsbRequest().also {
            if (!it.initialize(connection, endpoint)) {
                it.close()
                throw PtpException("Camera event endpoint could not be opened.")
            }
            eventRequest = it
        }
        fun queue() {
            eventBuffer.clear()
            if (!request.queue(eventBuffer)) throw PtpException("Camera event request could not be queued.")
            eventQueued = true
        }
        if (!eventQueued) queue()
        // Keep the request queued after a timeout; cancelling it would lose events between polls.
        val completed = try { connection.requestWait(timeoutMs.coerceAtLeast(1).toLong()) }
            catch (_: TimeoutException) { return null }
        if (completed !== request) throw PtpException("Camera event read failed.")
        eventQueued = false
        eventBuffer.flip()
        val bytes = ByteArray(eventBuffer.remaining()).also(eventBuffer::get)
        queue()
        return bytes.takeIf { it.isNotEmpty() }
    }

    override fun close() {
        if (closed) return
        closed = true
        try {
            try {
                eventRequest?.let { request ->
                    try { if (eventQueued) request.cancel() } finally { request.close() }
                }
            } finally {
                try { connection.releaseInterface(iface) } finally { connection.close() }
            }
        } finally { ownership.release() }
    }

    private companion object {
        // ponytail: one camera per phone; use per-device leases if multi-camera is introduced.
        val ownership = Semaphore(1, true)
    }
}
