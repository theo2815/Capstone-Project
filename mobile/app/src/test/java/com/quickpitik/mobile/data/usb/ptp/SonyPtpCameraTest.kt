package com.quickpitik.mobile.data.usb.ptp

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SonyPtpCameraTest {
    @Test fun `version 3 handshake reads successive images from the same buffer without control writes`() {
        val script = Script().apply {
            setup()
            packet(SonyPtpCamera.ALL_PROPERTIES, values(pending = 0x8000))
            repeat(2) {
                packet(SonyPtpCamera.ALL_PROPERTIES, values())
                packet(Ptp.OP_GET_OBJECT_INFO, objectInfo())
                packet(Ptp.OP_GET_OBJECT, byteArrayOf(1, 2, (3 + it).toByte()))
            }
        }
        val camera = camera(script.transport())
        camera.startLive()
        assertTrue(camera.capabilities.transientObjects)
        assertNull(camera.nextCapture())
        repeat(2) {
            val (handle, info) = requireNotNull(camera.nextCapture())
            assertEquals(SonyPtpCamera.BUFFER_HANDLE, handle)
            val out = ByteArrayOutputStream()
            assertEquals(3L, camera.readObject(handle, out, info.expectedSize))
            assertArrayEquals(byteArrayOf(1, 2, (3 + it).toByte()), out.toByteArray())
        }
        val commands = script.transport.writes.map { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN) }
        assertEquals(listOf(1, 2, 300, 3), commands.take(4).map { it.getInt(12) })
        assertEquals(1, commands[2].getInt(16))
        assertTrue(commands.all { (it.getShort(6).toInt() and 0xffff) in setOf(
            SonyPtpCamera.CONNECT, SonyPtpCamera.EXT_INFO, SonyPtpCamera.ALL_PROPERTIES,
            Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT) })
    }

    @Test fun `version 3 clean rejection falls back to version 2 with its one-parameter request`() {
        val script = Script().apply {
            packet(SonyPtpCamera.CONNECT); packet(SonyPtpCamera.CONNECT)
            reject(Ptp.RC_OPERATION_NOT_SUPPORTED)
            packet(SonyPtpCamera.EXT_INFO, extension(200))
            packet(SonyPtpCamera.CONNECT); packet(SonyPtpCamera.ALL_PROPERTIES, values())
        }
        val camera = camera(script.transport())
        camera.startLive()
        assertTrue(camera.capabilities.physicalShutter.contains("protocol 2"))
        assertEquals(16, script.transport.writes[3].size)
        assertEquals(200, ByteBuffer.wrap(script.transport.writes[3]).order(ByteOrder.LITTLE_ENDIAN).getInt(12))
    }

    @Test fun `returned version 2 and advertised single-property reads support older bodies`() {
        val script = Script().apply {
            packet(SonyPtpCamera.CONNECT); packet(SonyPtpCamera.CONNECT)
            packet(SonyPtpCamera.EXT_INFO, extension(200, SonyPtpCamera.PROPERTY))
            packet(SonyPtpCamera.CONNECT)
            for ((code, value) in listOf(SonyPtpCamera.DESTINATION to 0x11,
                SonyPtpCamera.OBJECT_IN_MEMORY to 0x8000, SonyPtpCamera.PRIORITY to 0))
                packet(SonyPtpCamera.PROPERTY, descriptor(code, value))
        }
        val camera = camera(script.transport())
        camera.startLive()
        assertTrue(camera.capabilities.transientObjects)
        assertEquals(listOf(SonyPtpCamera.DESTINATION, SonyPtpCamera.OBJECT_IN_MEMORY, SonyPtpCamera.PRIORITY),
            script.transport.writes.drop(4).map { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).getInt(12) })
    }

    @Test fun `response-only connect and temporarily empty capabilities preserve handshake framing`() {
        val script = Script().apply {
            reject(Ptp.RC_OK); reject(Ptp.RC_OK); reject(Ptp.RC_OK)
            packet(SonyPtpCamera.EXT_INFO, extension(300)); reject(Ptp.RC_OK)
            packet(SonyPtpCamera.ALL_PROPERTIES, values())
        }
        camera(script.transport()).startLive()
        assertEquals(6, script.transport.writes.size)
        val empty = Script().apply { repeat(5) { reject(Ptp.RC_OK) } }
        assertThrows(CameraSetupException::class.java) { camera(empty.transport()).startLive() }
        assertEquals(5, empty.transport.writes.size)
    }

    @Test fun `PC-only storage and application priority fail before consuming an image`() {
        for ((destination, priority) in listOf(1 to 0, 0x10 to 0, 0x11 to 1)) {
            val script = Script().apply { setup(values(destination = destination, priority = priority)) }
            val camera = camera(script.transport())
            assertThrows(CameraSetupException::class.java) { camera.startLive() }
            assertFalse(camera.capabilities.transientObjects)
            assertEquals(5, script.transport.writes.size)
        }
    }

    @Test fun `changed destination and RAW buffers stop before GetObject`() {
        val changed = Script().apply { setup(); packet(SonyPtpCamera.ALL_PROPERTIES, values(destination = 1)) }
        val camera = camera(changed.transport()).apply { startLive() }
        assertThrows(CameraSetupException::class.java) { camera.nextCapture() }
        val raw = Script().apply {
            setup(); packet(SonyPtpCamera.ALL_PROPERTIES, values())
            packet(Ptp.OP_GET_OBJECT_INFO, objectInfo(0xb101))
        }
        val rawCamera = camera(raw.transport()).apply { startLive() }
        assertThrows(CameraSetupException::class.java) { rawCamera.nextCapture() }
        assertEquals(7, raw.transport.writes.size)
    }

    @Test fun `datasets handle secondary enums and bound every truncated prefix`() {
        val enum = bytes {
            putShort(SonyPtpCamera.DESTINATION.toShort()); putShort(4); put(1); put(1)
            putShort(0x11); putShort(0x11); put(2)
            putShort(2); putShort(1); putShort(0x11)
            putShort(1); putShort(0x11)
        }
        val properties = bytes { putInt(2); putInt(0); put(enum); put(descriptor(SonyPtpCamera.OBJECT_IN_MEMORY, 0x8001)) }
        assertEquals(0x11L, SonyPtpData.properties(properties)[SonyPtpCamera.DESTINATION]?.value)
        for (dataset in listOf(extension(300), properties)) {
            for (length in 0 until dataset.size) {
                assertThrows(PtpException::class.java) {
                    if (dataset === properties) SonyPtpData.properties(dataset.copyOf(length))
                    else SonyPtpData.extension(dataset.copyOf(length))
                }
            }
        }
        assertThrows(PtpException::class.java) { SonyPtpData.properties(bytes { putInt(Int.MAX_VALUE); putInt(0) }) }
        assertThrows(PtpException::class.java) { SonyPtpData.property(enum, SonyPtpCamera.PRIORITY) }
        assertThrows(PtpException::class.java) { SonyPtpData.extension(extension(300) + byteArrayOf(0)) }
    }

    @Test fun `non-scalar properties do not desynchronize following scalar descriptors`() {
        val data = bytes {
            putInt(3); putInt(0)
            putShort(0xd200.toShort()); putShort(0xffff.toShort()); put(0); put(1)
            repeat(2) { put(2); putShort(65); putShort(0) }; put(0)
            putShort(0xd201.toShort()); putShort(0x4004); put(0); put(1)
            repeat(2) { putInt(2); putShort(1); putShort(2) }; put(0)
            put(descriptor(SonyPtpCamera.OBJECT_IN_MEMORY, 0x8001))
        }
        val parsed = SonyPtpData.properties(data)
        assertNull(parsed[0xd200]?.value)
        assertNull(parsed[0xd201]?.value)
        assertEquals(0x8001L, parsed[SonyPtpCamera.OBJECT_IN_MEMORY]?.value)
    }

    private fun camera(transport: ScriptedPtp) = SonyPtpCamera(PtpSession(transport), PtpDeviceInfo(
        100, 6, 100, "Sony", listOf(SonyPtpCamera.CONNECT, SonyPtpCamera.EXT_INFO),
        emptyList(), emptyList(), emptyList(), emptyList(), "Sony", "Test", "1", "serial"), "camera")

    private inner class Script {
        private val incoming = mutableListOf<ByteArray>()
        private var tx = 0
        lateinit var transport: ScriptedPtp
        fun packet(op: Int, payload: ByteArray = byteArrayOf()) {
            incoming += PtpWire.container(Ptp.TYPE_DATA, op, tx, payload)
            incoming += PtpWire.container(Ptp.TYPE_RESPONSE, Ptp.RC_OK, tx++)
        }
        fun reject(code: Int) { incoming += PtpWire.container(Ptp.TYPE_RESPONSE, code, tx++) }
        fun setup(properties: ByteArray = values()) {
            packet(SonyPtpCamera.CONNECT); packet(SonyPtpCamera.CONNECT)
            packet(SonyPtpCamera.EXT_INFO, extension(300)); packet(SonyPtpCamera.CONNECT)
            packet(SonyPtpCamera.ALL_PROPERTIES, properties)
        }
        fun transport() = ScriptedPtp(*incoming.toTypedArray()).also { transport = it }
    }

    private fun extension(version: Int, propertyOp: Int = SonyPtpCamera.ALL_PROPERTIES) = bytes {
        putShort(version.toShort()); putInt(3)
        listOf(Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT, propertyOp).forEach { putShort(it.toShort()) }
        putInt(3)
        listOf(SonyPtpCamera.DESTINATION, SonyPtpCamera.OBJECT_IN_MEMORY, SonyPtpCamera.PRIORITY).forEach { putShort(it.toShort()) }
    }
    private fun values(destination: Int = 0x11, pending: Int = 0x8001, priority: Int = 0) = bytes {
        putInt(3); putInt(0)
        put(descriptor(SonyPtpCamera.DESTINATION, destination))
        put(descriptor(SonyPtpCamera.OBJECT_IN_MEMORY, pending)); put(descriptor(SonyPtpCamera.PRIORITY, priority))
    }
    private fun descriptor(code: Int, value: Int) = bytes {
        putShort(code.toShort()); putShort(4); put(0); put(1)
        putShort(value.toShort()); putShort(value.toShort()); put(0)
    }
    private fun objectInfo(format: Int = 0x3801) = ByteArray(53).also {
        ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).putShort(4, format.toShort()).putInt(8, 3)
    }
    private fun bytes(write: ByteBuffer.() -> Unit) = ByteBuffer.allocate(512).order(ByteOrder.LITTLE_ENDIAN)
        .apply(write).let { it.array().copyOf(it.position()) }
}
