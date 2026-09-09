package com.quickpitik.mobile.data.usb.ptp

import java.io.ByteArrayOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Shared framing for metadata and streamed files. A broken transaction poisons the session. */
internal class PtpWire(
    private val transport: PtpTransport,
    private val waitMs: (Long) -> Unit = Thread::sleep,
) {
    var usable = true
        private set
    var transactionId = 0
    var checkCancelled: () -> Unit = {}
    var allowUnknownObjectLength = false
    private var pending = ByteArray(0)

    fun response(op: Int, vararg params: Int): Int = transaction {
        var retries = 0
        var code: Int
        do { code = readResponse(command(op, params)) }
        while (retryBusy(op, code, retries++))
        code
    }

    fun sendData(op: Int, payload: ByteArray, vararg params: Int) = transaction {
        val tx = command(op, params)
        write(container(Ptp.TYPE_DATA, op, tx, payload))
        requireOk(readResponse(tx))
    }

    fun data(op: Int, vararg params: Int): ByteArray {
        val out = ByteArrayOutputStream()
        receive(op, params, out, MAX_METADATA, null, false)
        return out.toByteArray()
    }

    fun objectTo(op: Int, handle: Long, sink: OutputStream, expectedSize: Long?): Long =
        receive(op, intArrayOf(handle.toInt()), sink, MAX_OBJECT, expectedSize, allowUnknownObjectLength)

    private fun receive(
        op: Int, params: IntArray, sink: OutputStream, limit: Long,
        expectedSize: Long?, allowUnknown: Boolean,
    ): Long = transaction {
        var tx: Int
        var header: Header
        var retries = 0
        do {
            tx = command(op, params)
            header = header(tx)
            if (header.type == Ptp.TYPE_RESPONSE) responsePayload(header)
        } while (header.type == Ptp.TYPE_RESPONSE && retryBusy(op, header.code, retries++))
        if (header.type == Ptp.TYPE_RESPONSE) {
            requireOk(header.code)
            // These polls/handshake stages may have no payload. Object and property reads may not.
            if (op == Ptp.OP_EOS_GET_EVENT || op == SonyPtpCamera.CONNECT || op == SonyPtpCamera.EXT_INFO)
                return@transaction 0L
            throw PtpException("Camera returned no data for op 0x%04X".format(op))
        }
        if (header.type != Ptp.TYPE_DATA || header.code != op) throw PtpException("Unexpected PTP data phase.")
        val count = payload(header, sink, limit, expectedSize, allowUnknown)
        requireOk(readResponse(tx))
        if (expectedSize != null && count != expectedSize) throw PtpException("Incomplete camera file: $count of $expectedSize bytes.")
        count
    }

    private data class Header(val size: Long, val type: Int, val code: Int)
    private var lastReadSize = 0

    private fun header(tx: Int): Header {
        val bytes = ByteArray(12)
        var have = 0
        var emptyReads = 0
        while (have < bytes.size) {
            if (pending.isEmpty()) {
                read()
                // A zero-length packet may terminate the preceding data phase.
                if (pending.isEmpty() && ++emptyReads > 1) throw PtpException("Missing PTP header.")
            }
            val n = minOf(bytes.size - have, pending.size)
            pending.copyInto(bytes, have, 0, n)
            pending = pending.copyOfRange(n, pending.size)
            have += n
        }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val size = bb.int.toLong() and 0xffffffffL
        val type = bb.short.toInt() and 0xffff
        val code = bb.short.toInt() and 0xffff
        if (bb.int != tx) throw PtpException("PTP transaction mismatch; reconnecting is required.")
        if (size < 12) throw PtpException("Invalid PTP container length: $size.")
        return Header(size, type, code)
    }

    private fun payload(h: Header, sink: OutputStream, limit: Long, expected: Long?, allowUnknown: Boolean): Long {
        val unknown = h.size == 0xffffffffL
        if (unknown && !allowUnknown) throw PtpException("Unexpected unknown PTP length.")
        val length = if (unknown) expected else h.size - 12
        if (length != null && length > limit) throw PtpException("Camera payload exceeds the transfer limit.")
        var written = 0L
        while (true) {
            checkCancelled()
            if (length != null && written == length) break
            if (pending.isEmpty()) {
                // Some cameras send the DATA header as its own short transfer.
                if (unknown && length == null && written > 0 && lastReadSize < CHUNK) break
                read()
                if (pending.isEmpty()) {
                    if (unknown && length == null) break
                    throw PtpException("Truncated camera payload ($written bytes).")
                }
            }
            val n = if (length == null) pending.size else minOf(pending.size.toLong(), length - written).toInt()
            if (written + n > limit) throw PtpException("Camera payload exceeds the transfer limit.")
            sink.write(pending, 0, n)
            written += n
            pending = pending.copyOfRange(n, pending.size)
        }
        return written
    }

    private fun readResponse(tx: Int): Int {
        val h = header(tx)
        if (h.type != Ptp.TYPE_RESPONSE) throw PtpException("Expected a PTP response.")
        responsePayload(h)
        return h.code
    }

    private fun responsePayload(h: Header) {
        if (h.size !in 12L..32L || (h.size - 12) % 4 != 0L)
            throw PtpException("Invalid PTP response parameters.")
        payload(h, ByteArrayOutputStream(), 20, null, false)
    }

    private fun retryBusy(op: Int, code: Int, retries: Int): Boolean {
        if (code != Ptp.RC_DEVICE_BUSY || op !in BUSY_RETRY_OPS || retries >= 3) return false
        // Only reached after a complete response BEFORE any DATA. Never replay
        // partial files, ambiguous USB failures, or non-idempotent shutter commands.
        repeat((retries + 1) * 4) { checkCancelled(); waitMs(50) }
        checkCancelled()
        return true
    }

    private fun command(op: Int, params: IntArray): Int {
        // OpenSession always uses transaction zero, including a clean Busy retry.
        val tx = if (op == Ptp.OP_OPEN_SESSION) 0 else transactionId
        transactionId = tx + 1
        val bytes = ByteBuffer.allocate(params.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        params.forEach(bytes::putInt)
        write(container(Ptp.TYPE_COMMAND, op, tx, bytes.array()))
        return tx
    }

    private fun read() {
        checkCancelled()
        val buffer = ByteArray(CHUNK)
        val n = transport.read(buffer, TIMEOUT_MS)
        if (n < 0 || n > buffer.size) throw PtpException("Camera USB read failed; reconnecting is required.")
        lastReadSize = n
        pending = buffer.copyOf(n)
    }

    private fun write(bytes: ByteArray) {
        checkCancelled()
        if (transport.write(bytes, TIMEOUT_MS) != bytes.size) throw PtpException("Incomplete USB command write.")
    }

    private inline fun <T> transaction(block: () -> T): T {
        if (!usable) throw PtpException("PTP session is invalid; reopen the camera.")
        return try { block() } catch (e: Throwable) {
            if (e !is PtpResponseException) usable = false
            throw e
        }
    }

    companion object {
        const val CHUNK = 16384
        const val TIMEOUT_MS = 5000
        const val MAX_METADATA = 16L * 1024 * 1024
        const val MAX_OBJECT = 1024L * 1024 * 1024
        private val BUSY_RETRY_OPS = setOf(
            Ptp.OP_GET_DEVICE_INFO, Ptp.OP_OPEN_SESSION, Ptp.OP_GET_STORAGE_IDS,
            Ptp.OP_GET_OBJECT_HANDLES, Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT, Ptp.OP_GET_THUMB,
            0x1014, 0x1015, // Standard GetDevicePropDesc / GetDevicePropValue.
            Ptp.OP_EOS_SET_REMOTE_MODE, Ptp.OP_EOS_SET_EVENT_MODE,
            Ptp.OP_EOS_KEEP_DEVICE_ON, Ptp.OP_EOS_GET_EVENT,
            Ptp.OP_NIKON_GET_EVENT, Ptp.OP_NIKON_GET_EVENT_EX,
            SonyPtpCamera.CONNECT, SonyPtpCamera.EXT_INFO, SonyPtpCamera.PROPERTY, SonyPtpCamera.ALL_PROPERTIES,
        )
        fun requireOk(code: Int) { if (code != Ptp.RC_OK) throw PtpResponseException(code) }
        fun container(type: Int, code: Int, tx: Int, payload: ByteArray = byteArrayOf()): ByteArray =
            ByteBuffer.allocate(12 + payload.size).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(12 + payload.size).putShort(type.toShort()).putShort(code.toShort()).putInt(tx).put(payload).array()
    }
}
