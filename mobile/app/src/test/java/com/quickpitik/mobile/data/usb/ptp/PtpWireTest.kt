package com.quickpitik.mobile.data.usb.ptp

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal class ScriptedPtp(vararg incoming: ByteArray) : PtpTransport {
    val reads = ArrayDeque(incoming.toList())
    val writes = mutableListOf<ByteArray>()
    var shortWrite = false
    var closed = false
    override fun read(buffer: ByteArray, timeoutMs: Int): Int {
        val bytes = reads.removeFirstOrNull() ?: return -1
        bytes.copyInto(buffer)
        return bytes.size
    }
    override fun write(buffer: ByteArray, timeoutMs: Int): Int {
        writes.add(buffer.copyOf())
        return if (shortWrite) buffer.size - 1 else buffer.size
    }
    override fun close() { closed = true }
}

class PtpWireTest {
    private fun data(bytes: ByteArray, tx: Int = 0) = PtpWire.container(Ptp.TYPE_DATA, Ptp.OP_GET_OBJECT, tx, bytes)
    private fun ok(tx: Int = 0) = PtpWire.container(Ptp.TYPE_RESPONSE, Ptp.RC_OK, tx)

    @Test fun `fragmented headers and coalesced response preserve exact bytes`() {
        val bytes = byteArrayOf(1, 2, 3, 4, 5)
        val all = data(bytes) + ok()
        val transport = ScriptedPtp(all.copyOfRange(0, 3), all.copyOfRange(3, 10), all.copyOfRange(10, all.size))
        val wire = PtpWire(transport)
        val out = ByteArrayOutputStream()
        assertEquals(5L, wire.objectTo(Ptp.OP_GET_OBJECT, 10, out, 5))
        assertArrayEquals(bytes, out.toByteArray())
        assertTrue(wire.usable)
    }

    @Test fun `transaction mismatch invalidates session and prevents later writes`() {
        val transport = ScriptedPtp(data(byteArrayOf(1), tx = 9))
        val wire = PtpWire(transport)
        assertThrows(PtpException::class.java) { wire.objectTo(Ptp.OP_GET_OBJECT, 1, ByteArrayOutputStream(), 1) }
        assertThrows(PtpException::class.java) { wire.response(Ptp.OP_GET_STORAGE_IDS) }
        assertEquals(1, transport.writes.size)
    }

    @Test fun `short command write is a failed transaction`() {
        val wire = PtpWire(ScriptedPtp().apply { shortWrite = true })
        assertThrows(PtpException::class.java) { wire.response(Ptp.OP_OPEN_SESSION, 1) }
        assertFalse(wire.usable)
    }

    @Test fun `truncation never returns a successful partial file`() {
        val transport = ScriptedPtp(data(byteArrayOf(1, 2, 3, 4)).dropLast(2).toByteArray())
        val wire = PtpWire(transport)
        assertThrows(PtpException::class.java) { wire.objectTo(Ptp.OP_GET_OBJECT, 1, ByteArrayOutputStream(), 4) }
        assertFalse(wire.usable)
    }

    @Test fun `expected object size is checked even after OK`() {
        val wire = PtpWire(ScriptedPtp(data(byteArrayOf(1, 2)), ok()))
        assertThrows(PtpException::class.java) { wire.objectTo(Ptp.OP_GET_OBJECT, 1, ByteArrayOutputStream(), 3) }
        assertFalse(wire.usable)
    }

    @Test fun `sink failure poisons stream and requires reopen`() {
        val wire = PtpWire(ScriptedPtp(data(byteArrayOf(1, 2)), ok()))
        val brokenDisk = object : OutputStream() { override fun write(b: Int) { throw IOException("disk full") } }
        assertThrows(IOException::class.java) { wire.objectTo(Ptp.OP_GET_OBJECT, 1, brokenDisk, 2) }
        assertFalse(wire.usable)
    }

    @Test fun `cancellation after payload cannot reuse the unfinished transaction`() {
        val transport = ScriptedPtp(data(byteArrayOf(1, 2)), ok())
        val wire = PtpWire(transport)
        val cancelAfterWrite = object : ByteArrayOutputStream() {
            override fun write(bytes: ByteArray, off: Int, len: Int) {
                super.write(bytes, off, len)
                wire.checkCancelled = { throw kotlinx.coroutines.CancellationException("Stopped") }
            }
        }
        assertThrows(kotlinx.coroutines.CancellationException::class.java) {
            wire.objectTo(Ptp.OP_GET_OBJECT, 1, cancelAfterWrite, 2)
        }
        wire.checkCancelled = {}
        assertThrows(PtpException::class.java) { wire.response(Ptp.OP_CLOSE_SESSION) }
        assertEquals(1, transport.writes.size)
    }

    @Test fun `clean unsupported response allows another transaction`() {
        val transport = ScriptedPtp(PtpWire.container(Ptp.TYPE_RESPONSE, 0x2005, 0), ok(1))
        val wire = PtpWire(transport)
        assertThrows(PtpResponseException::class.java) { wire.data(Ptp.OP_GET_OBJECT_INFO, 1) }
        assertTrue(wire.usable)
        assertEquals(Ptp.RC_OK, wire.response(Ptp.OP_CLOSE_SESSION))
    }

    @Test fun `unknown length is opt in and bounded by expected file size`() {
        val unknown = data(byteArrayOf(1, 2, 3))
        ByteBuffer.wrap(unknown).order(ByteOrder.LITTLE_ENDIAN).putInt(-1)
        val rejected = PtpWire(ScriptedPtp(unknown, ok()))
        assertThrows(PtpException::class.java) { rejected.objectTo(Ptp.OP_GET_OBJECT, 1, ByteArrayOutputStream(), 3) }
        val accepted = PtpWire(ScriptedPtp(unknown, ok())).apply { allowUnknownObjectLength = true }
        assertEquals(3L, accepted.objectTo(Ptp.OP_GET_OBJECT, 1, ByteArrayOutputStream(), 3))
    }

    @Test fun `oversized and invalid lengths fail before allocating payload`() {
        for (length in listOf(5, Int.MAX_VALUE)) {
            val packet = data(byteArrayOf())
            ByteBuffer.wrap(packet).order(ByteOrder.LITTLE_ENDIAN).putInt(length)
            val wire = PtpWire(ScriptedPtp(packet))
            assertThrows(PtpException::class.java) { wire.objectTo(Ptp.OP_GET_OBJECT, 1, ByteArrayOutputStream(), null) }
        }
    }

    @Test fun `closing unopened session does not read or issue close command`() {
        val transport = ScriptedPtp()
        PtpSession(transport).close()
        assertTrue(transport.closed)
        assertTrue(transport.writes.isEmpty())
    }
}
