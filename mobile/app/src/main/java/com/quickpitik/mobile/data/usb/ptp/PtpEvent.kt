package com.quickpitik.mobile.data.usb.ptp

import java.nio.BufferUnderflowException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Event transaction IDs need not match the current command; events use a separate USB pipe. */
internal data class PtpEvent(val code: Int, val params: List<Long>) {
    val handle: Long? get() = params.firstOrNull()?.takeIf { it != 0L && it != 0xffffffffL }

    companion object {
        fun parseUsb(bytes: ByteArray): List<PtpEvent> = parse(bytes) { bb ->
            buildList {
                while (bb.hasRemaining()) {
                    val size = bb.int
                    if (size !in 12..24 || size % 4 != 0 || size - 4 > bb.remaining())
                        throw PtpException("Invalid USB camera event length.")
                    if (bb.u16() != Ptp.TYPE_EVENT) throw PtpException("Invalid USB camera event type.")
                    val code = bb.u16()
                    bb.int // asynchronous transaction identifier
                    add(PtpEvent(code, List((size - 12) / 4) { bb.u32() }))
                }
            }
        }

        fun parseNikon(bytes: ByteArray, extended: Boolean): List<PtpEvent> = parse(bytes) { bb ->
            val count = bb.u16()
            if (extended) bb.u16() // reserved
            val minimum = if (extended) 4 else 6
            if (count > bb.remaining() / minimum) throw PtpException("Invalid Nikon event count.")
            val events = List(count) {
                val code = bb.u16()
                val parameters = if (extended) bb.u16() else 1
                if (parameters > 5 || parameters > bb.remaining() / 4)
                    throw PtpException("Invalid Nikon event parameters.")
                PtpEvent(code, List(parameters) { bb.u32() })
            }
            if (bb.hasRemaining()) throw PtpException("Unexpected Nikon event data.")
            events
        }

        private fun <T> parse(bytes: ByteArray, read: (ByteBuffer) -> T): T = try {
            read(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN))
        } catch (e: BufferUnderflowException) {
            throw PtpException("Truncated camera event.", e)
        }
        private fun ByteBuffer.u16() = short.toInt() and 0xffff
        private fun ByteBuffer.u32() = int.toLong() and 0xffffffffL
    }
}
