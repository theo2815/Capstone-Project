package com.quickpitik.mobile.data.usb.ptp

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class CameraDriverTest {
    private fun info(manufacturer: String, operations: List<Int>) = PtpDeviceInfo(
        100, 6, 100, "MTP", operations, emptyList(), emptyList(), emptyList(), listOf(0x3801),
        manufacturer, "Test camera", "1.0", "serial",
    )

    @Test fun `Microsoft extension ID does not hide Canon EOS capability`() {
        val info = info("Canon", listOf(Ptp.OP_EOS_SET_REMOTE_MODE, Ptp.OP_EOS_SET_EVENT_MODE))
        assertTrue(CameraDrivers.isCanon(0x04a9, info))
        assertFalse(CameraDrivers.isCanon(0x054c, info.copy(manufacturer = "Sony")))
        assertFalse(CameraDrivers.isCanon(0x04a9, info.copy(operationsSupported = emptyList())))
    }

    @Test fun `generic live initialization sends no vendor operations`() {
        val transport = ScriptedPtp()
        GenericPtpCamera(PtpSession(transport), info("Nikon", emptyList()), "camera").startLive()
        assertTrue(transport.writes.isEmpty())
    }

    @Test fun `R10 warmup and card polling quirks are narrow`() {
        assertTrue(CameraDrivers.needsWarmUp(0x04a9, 0x32f8))
        assertFalse(CameraDrivers.needsWarmUp(0x04a9, 0x32f5))
        assertFalse(CameraDrivers.needsWarmUp(0x054c, 0x32f8))
        assertTrue(CameraDrivers.cardPollingOnly("Canon EOS R6"))
        assertFalse(CameraDrivers.cardPollingOnly("Canon EOS R6 Mark II"))
    }

    @Test fun `Canon browse session enters remote mode then re-reads capabilities without event mode`() {
        val transport = ScriptedPtp(ok(0), data(Ptp.OP_GET_DEVICE_INFO, 1, deviceInfoBytes("Canon EOS R6")), ok(1))
        val camera = canon(PtpSession(transport))
        camera.startSession()
        assertEquals(listOf(Ptp.OP_EOS_SET_REMOTE_MODE to 1, Ptp.OP_GET_DEVICE_INFO to null), transport.commands())
        assertEquals("Canon EOS R6", camera.info.model)
    }

    @Test fun `quirk table mirrors libgphoto2 for Canon families and Nikon 1`() {
        assertEquals(0x15, CameraQuirks.forCamera(0x04a9, 0x32c5, "Canon EOS M50").eosRemoteMode)
        assertEquals(1, CameraQuirks.forCamera(0x04a9, 0, "Canon EOS M6 Mark II").eosRemoteMode)
        assertEquals(0x11, CameraQuirks.forCamera(0x04a9, 0, "Canon PowerShot SX720 HS").eosRemoteMode)
        assertEquals(CameraQuirks(), CameraQuirks.forCamera(0x054c, 0x0da7, "ILCE-7M4"))
        assertFalse(CameraQuirks.forCamera(0, 0, "Canon EOS R6").vendorEvents)
        assertTrue(CameraQuirks.forCamera(0, 0, "Canon EOS R6 Mark II").vendorEvents)
        assertTrue(CameraQuirks.forCamera(0x04a9, 0x32f8, "").warmUp)
        assertFalse(CameraQuirks.forCamera(0, 0, "NIKON 1 J4").vendorEvents)
        val transport = ScriptedPtp(ok(0), data(Ptp.OP_GET_DEVICE_INFO, 1, deviceInfoBytes("Canon EOS M50")), ok(1))
        canon(PtpSession(transport), CameraQuirks.forCamera(0x04a9, 0x32c5, "Canon EOS M50")).startSession()
        assertEquals(Ptp.OP_EOS_SET_REMOTE_MODE to 0x15, transport.commands().first())
    }

    @Test fun `Canon close hands the body back in remote mode one`() {
        val transport = ScriptedPtp(ok(0), ok(1), data(Ptp.OP_GET_DEVICE_INFO, 2, deviceInfoBytes("Canon EOS R6")), ok(2),
            ok(3), ok(4), ok(5), ok(6), ok(7))
        val session = PtpSession(transport)
        val camera = canon(session)
        session.openSession()
        camera.startSession()
        camera.startLive()
        camera.close()
        assertEquals(
            listOf(
                Ptp.OP_OPEN_SESSION to 1, Ptp.OP_EOS_SET_REMOTE_MODE to 1, Ptp.OP_GET_DEVICE_INFO to null, Ptp.OP_EOS_SET_EVENT_MODE to 1,
                Ptp.OP_EOS_SET_EVENT_MODE to 0, Ptp.OP_EOS_SET_REMOTE_MODE to 0, Ptp.OP_EOS_SET_REMOTE_MODE to 1,
                Ptp.OP_CLOSE_SESSION to null,
            ),
            transport.commands(),
        )
        assertTrue(transport.closed)
    }

    @Test fun `Fujifilm priority is handed back to the camera only when it reads USB`() {
        val fuji = info("FUJIFILM", listOf(Ptp.OP_GET_OBJECT_HANDLES, Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT))
            .copy(devicePropertiesSupported = listOf(Ptp.FUJI_DPC_PRIORITY_MODE))
        val usb = ScriptedPtp(data(Ptp.OP_GET_DEVICE_PROP_VALUE, 0, byteArrayOf(2, 0)), ok(0), ok(1))
        FujiPtpCamera(PtpSession(usb), fuji, "camera").startSession()
        assertEquals(listOf(Ptp.OP_GET_DEVICE_PROP_VALUE to Ptp.FUJI_DPC_PRIORITY_MODE,
            Ptp.OP_SET_DEVICE_PROP_VALUE to Ptp.FUJI_DPC_PRIORITY_MODE), usb.commands())
        assertArrayEquals(byteArrayOf(1, 0), usb.writes.last().copyOfRange(12, 14))
        val camera = ScriptedPtp(data(Ptp.OP_GET_DEVICE_PROP_VALUE, 0, byteArrayOf(1, 0)), ok(0))
        FujiPtpCamera(PtpSession(camera), fuji, "camera").startSession()
        assertEquals(1, camera.commands().size)
        val silent = ScriptedPtp()
        FujiPtpCamera(PtpSession(silent), fuji.copy(devicePropertiesSupported = emptyList()), "camera").startSession()
        assertTrue(silent.writes.isEmpty())
    }

    @Test fun `vendor object-added events are accepted only by their adapter`() {
        fun transport(code: Int) = ScriptedPtp().apply { interrupts.add(PtpWire.container(Ptp.TYPE_EVENT, code, 0, words(42))) }
        assertEquals(listOf(42L), PanasonicPtpCamera(PtpSession(transport(Ptp.EV_PANASONIC_OBJECT_ADDED_SDRAM)), info("Panasonic", emptyList()), "camera").events())
        assertEquals(listOf(42L), FujiPtpCamera(PtpSession(transport(Ptp.EV_FUJI_OBJECT_ADDED)), info("FUJIFILM", emptyList()), "camera").events())
        assertTrue(GenericPtpCamera(PtpSession(transport(Ptp.EV_PANASONIC_OBJECT_ADDED_SDRAM)), info("Other", emptyList()), "camera").events().isEmpty())
        assertEquals(listOf(42L), GenericPtpCamera(PtpSession(transport(Ptp.EV_OBJECT_ADDED)), info("Other", emptyList()), "camera").events())
    }

    @Test fun `object identity separates cameras folders and reused handles`() {
        val photo = PtpObjectInfo(0x3801, 100, "IMG_0001.JPG", 1, 10, "20260907T120000")
        assertNotEquals(photo.sourceKey("camera-a", 1), photo.sourceKey("camera-b", 1))
        assertNotEquals(photo.sourceKey("camera-a", 1), photo.copy(parent = 11).sourceKey("camera-a", 1))
        assertNotEquals(photo.sourceKey("camera-a", 1), photo.copy(capturedAt = "20260907T120001").sourceKey("camera-a", 1))
    }

    @Test fun `truncated metadata and hostile capability counts are rejected`() {
        assertThrows(PtpException::class.java) { PtpObjectInfo.parse(ByteArray(12)) }
        val payload = ByteBuffer.allocate(17).order(ByteOrder.LITTLE_ENDIAN)
            .putShort(100).putInt(6).putShort(100).put(0).putShort(0).putInt(Int.MAX_VALUE).array()
        assertThrows(PtpException::class.java) { PtpDeviceInfo.parse(payload) }
    }

    @Test fun `identification opens session before device info with distinct transactions`() {
        val dataset = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN).apply {
            putShort(100); putInt(6); putShort(100); put(0); putShort(0)
            repeat(5) { putInt(0) }
            for (value in listOf("Canon", "Canon EOS R6", "1.0", "serial")) {
                put((value.length + 1).toByte())
                value.forEach { putShort(it.code.toShort()) }; putShort(0)
            }
        }.let { it.array().copyOf(it.position()) }
        val transport = ScriptedPtp(ok(0), data(Ptp.OP_GET_DEVICE_INFO, 1, dataset), ok(1))
        assertEquals("Canon EOS R6", PtpSession(transport).identify().model)
        val commands = transport.writes.map { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN) }
        assertEquals(Ptp.OP_OPEN_SESSION, commands[0].getShort(6).toInt())
        assertEquals(0, commands[0].getInt(8))
        assertEquals(Ptp.OP_GET_DEVICE_INFO, commands[1].getShort(6).toInt())
        assertEquals(1, commands[1].getInt(8))
    }

    @Test fun `empty second slot does not hide readable first card`() {
        val transport = ScriptedPtp(
            data(Ptp.OP_GET_STORAGE_IDS, 0, words(2, 1, 2)), ok(0),
            data(Ptp.OP_GET_OBJECT_HANDLES, 1, words(1, 42)), ok(1),
            response(Ptp.RC_STORE_NOT_AVAILABLE, 2),
        )
        assertEquals(listOf(42L), storageCamera(transport).handles())
    }

    @Test fun `all unavailable slots and transport errors never become an empty baseline`() {
        val unavailable = ScriptedPtp(data(Ptp.OP_GET_STORAGE_IDS, 0, words(1, 1)), ok(0),
            response(Ptp.RC_INVALID_STORAGE_ID, 1))
        assertThrows(PtpResponseException::class.java) { storageCamera(unavailable).handles() }
        val disconnected = ScriptedPtp(data(Ptp.OP_GET_STORAGE_IDS, 0, words(1, 1)), ok(0))
        assertThrows(PtpException::class.java) { storageCamera(disconnected).handles() }
    }

    @Test fun `storage-less enumeration uses standard all-stores wildcard`() {
        val transport = ScriptedPtp(data(Ptp.OP_GET_OBJECT_HANDLES, 0, words(1, 42)), ok(0))
        val camera = GenericPtpCamera(PtpSession(transport), info("Fuji", listOf(
            Ptp.OP_GET_OBJECT_HANDLES, Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT)), "camera")
        assertTrue(camera.capabilities.cardImport)
        assertEquals(listOf(42L), camera.handles())
        assertEquals(-1, ByteBuffer.wrap(transport.writes.single()).order(ByteOrder.LITTLE_ENDIAN).getInt(12))
    }

    @Test fun `Nikon extended event rejection falls back without changing control mode`() {
        val payload = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
            .putShort(1).putShort(Ptp.EV_OBJECT_ADDED.toShort()).putInt(42).array()
        val transport = ScriptedPtp(response(Ptp.RC_OPERATION_NOT_SUPPORTED, 0),
            data(Ptp.OP_NIKON_GET_EVENT, 1, payload), ok(1))
        val camera = NikonPtpCamera(PtpSession(transport), info("Nikon", listOf(
            Ptp.OP_NIKON_GET_EVENT, Ptp.OP_NIKON_GET_EVENT_EX)), "camera")
        assertEquals(listOf(42L), camera.events())
        assertEquals(listOf(Ptp.OP_NIKON_GET_EVENT_EX, Ptp.OP_NIKON_GET_EVENT),
            transport.writes.map { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).getShort(6).toInt() and 0xffff })
    }

    @Test fun `Nikon vendor event queue is used instead of the interrupt pipe when advertised`() {
        val extended = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
            .putShort(1).putShort(0).putShort(Ptp.EV_OBJECT_ADDED.toShort()).putShort(1).putInt(42).array()
        val transport = ScriptedPtp(data(Ptp.OP_NIKON_GET_EVENT_EX, 0, extended), ok(0))
            .apply { interrupts.add(PtpWire.container(Ptp.TYPE_EVENT, Ptp.EV_OBJECT_ADDED, 0, words(7))) }
        val camera = NikonPtpCamera(PtpSession(transport), info("Nikon", listOf(Ptp.OP_NIKON_GET_EVENT_EX)), "camera")
        assertEquals(listOf(42L), camera.events())
        assertEquals(1, transport.interrupts.size)
    }

    @Test fun `Nikon 1 avoids broken advertised vendor event commands`() {
        val transport = ScriptedPtp()
        val camera = NikonPtpCamera(PtpSession(transport), info("Nikon", listOf(Ptp.OP_NIKON_GET_EVENT))
            .copy(model = "NIKON 1 J4"), "camera")
        assertFalse(camera.capabilities.eventPolling)
        assertTrue(camera.events().isEmpty())
        assertTrue(transport.writes.isEmpty())
    }

    @Test fun `asynchronous and Nikon event parsers reject truncation and preserve unsigned handles`() {
        val usb = PtpWire.container(Ptp.TYPE_EVENT, Ptp.EV_OBJECT_ADDED, -1, words(-2))
        assertEquals(0xfffffffeL, PtpEvent.parseUsb(usb).single().handle)
        val extended = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
            .putShort(1).putShort(0).putShort(Ptp.EV_OBJECT_ADDED.toShort()).putShort(1).putInt(-2).array()
        assertEquals(0xfffffffeL, PtpEvent.parseNikon(extended, true).single().handle)
        for (length in 0 until extended.size)
            assertThrows(PtpException::class.java) { PtpEvent.parseNikon(extended.copyOf(length), true) }
        assertThrows(PtpException::class.java) { PtpEvent.parseUsb(usb.copyOf(usb.size - 1)) }
        assertThrows(PtpException::class.java) { PtpEvent.parseUsb(usb + byteArrayOf(1)) }
    }

    private fun storageCamera(transport: ScriptedPtp) = GenericPtpCamera(PtpSession(transport),
        info("Nikon", listOf(Ptp.OP_GET_STORAGE_IDS, Ptp.OP_GET_OBJECT_HANDLES, Ptp.OP_GET_OBJECT_INFO, Ptp.OP_GET_OBJECT)), "camera")
    private fun canon(session: PtpSession, quirks: CameraQuirks = CameraQuirks()) =
        CanonEosCamera(session, info("Canon", listOf(Ptp.OP_EOS_SET_REMOTE_MODE, Ptp.OP_EOS_SET_EVENT_MODE)), "camera", quirks)
    private fun deviceInfoBytes(model: String) = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN).apply {
        putShort(100); putInt(6); putShort(100); put(0); putShort(0)
        repeat(5) { putInt(0) }
        for (value in listOf("Canon", model, "1.0", "serial")) {
            put((value.length + 1).toByte())
            value.forEach { putShort(it.code.toShort()) }; putShort(0)
        }
    }.let { it.array().copyOf(it.position()) }
    /** (op, first param or null) per command container written, in wire order. */
    private fun ScriptedPtp.commands() = writes.map { ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN) }
        .filter { it.getShort(4).toInt() == Ptp.TYPE_COMMAND }
        .map { (it.getShort(6).toInt() and 0xffff) to (if (it.limit() >= 16) it.getInt(12) else null) }
    private fun words(vararg values: Int) = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        .apply { values.forEach { putInt(it) } }.array()
    private fun data(op: Int, tx: Int, bytes: ByteArray) = PtpWire.container(Ptp.TYPE_DATA, op, tx, bytes)
    private fun ok(tx: Int) = response(Ptp.RC_OK, tx)
    private fun response(code: Int, tx: Int) = PtpWire.container(Ptp.TYPE_RESPONSE, code, tx)
}
