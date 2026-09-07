package com.quickpitik.mobile.data.usb.ptp

import java.nio.BufferUnderflowException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Read-only SDIO ingest; never changes camera priority, save destination or exposure. */
class SonyPtpCamera(session: PtpSession, info: PtpDeviceInfo, cameraKey: String) : GenericPtpCamera(session, info, cameraKey) {
    private var operations = info.operationsSupported.toSet()
    private var properties = info.devicePropertiesSupported.toSet()
    private var version = 0
    private var ready = false
    override val capabilities get() = super.capabilities.copy(
        transientObjects = ready,
        physicalShutter = "Sony PC Remote protocol ${version / 100}: camera + PC JPEG copies; verify card backup and physical shutter before a shoot.",
    )

    override fun startLive() {
        session.vendorData(CONNECT, 1, 0, 0)
        session.vendorData(CONNECT, 2, 0, 0)
        var requestVersion = 300
        var extension: SonyPtpData.Extension? = null
        repeat(3) {
            if (extension != null) return@repeat
            val bytes = try {
                if (requestVersion == 300) session.vendorData(EXT_INFO, 300, 1)
                else session.vendorData(EXT_INFO, 200)
            } catch (e: PtpResponseException) {
                if (requestVersion != 300 || e.code !in setOf(Ptp.RC_OPERATION_NOT_SUPPORTED, 0x201d)) throw e
                requestVersion = 200
                session.vendorData(EXT_INFO, 200)
            }
            if (bytes.isNotEmpty()) SonyPtpData.extension(bytes).takeIf { it.codes.isNotEmpty() }?.let { extension = it }
        }
        val negotiated = extension ?: throw CameraSetupException("Sony did not provide PC Remote capabilities. Check USB mode and reconnect.")
        version = negotiated.version
        operations = operations + negotiated.codes.filter { (it and 0x7000) == 0x1000 }
        properties = properties + negotiated.codes.filter { (it and 0x7000) == 0x5000 }
        session.vendorData(CONNECT, 3, 0, 0)
        if (Ptp.OP_GET_OBJECT_INFO !in operations || Ptp.OP_GET_OBJECT !in operations ||
            (ALL_PROPERTIES !in operations && PROPERTY !in operations))
            throw CameraSetupException("This Sony PC Remote protocol cannot supply JPEG copies here. Use MTP card import.")
        validateDestination(readProperties())
        ready = true
    }

    override fun nextCapture(): Pair<Long, PtpObjectInfo>? {
        check(ready) { "Sony PC Remote is not initialized." }
        val values = readProperties()
        validateDestination(values)
        val pending = values[OBJECT_IN_MEMORY]
        if (pending?.type != 4 || pending.value == null)
            throw CameraSetupException("This Sony mode does not report pending JPEG transfers. Use MTP card import.")
        if (pending.value <= 0x8000) return null
        val info = session.getObjectInfo(BUFFER_HANDLE)
        if (!info.isJpeg || info.isAssociation)
            throw CameraSetupException("Select JPEG only for Sony PC transfers (RAW can remain on the card), then restart auto-upload.")
        if (info.expectedSize == null)
            throw CameraSetupException("Sony has not provided a complete JPEG size. Wait for the camera to finish saving and restart.")
        return BUFFER_HANDLE to info
    }

    private fun readProperties(): Map<Int, SonyPtpData.Property> = if (ALL_PROPERTIES in operations) {
        SonyPtpData.properties(session.vendorData(ALL_PROPERTIES))
    } else {
        listOf(DESTINATION, OBJECT_IN_MEMORY, PRIORITY).filter { it in properties }.associateWith {
            SonyPtpData.property(session.vendorData(PROPERTY, it), it)
        }
    }

    private fun validateDestination(values: Map<Int, SonyPtpData.Property>) {
        if (values[DESTINATION]?.value != 0x11L)
            throw CameraSetupException("On Sony, set Still Image Save Destination to PC + Camera before starting. This mode must retain a card backup.")
        if (values[PRIORITY]?.value?.let { it != 0L } == true)
            throw CameraSetupException("On Sony, set Priority to Camera so you can use the physical shutter, then start again.")
    }

    companion object {
        internal const val CONNECT = 0x9201
        internal const val EXT_INFO = 0x9202
        internal const val PROPERTY = 0x9203
        internal const val ALL_PROPERTIES = 0x9209
        internal const val OBJECT_IN_MEMORY = 0xd215
        internal const val DESTINATION = 0xd222
        internal const val PRIORITY = 0xd25a
        internal const val BUFFER_HANDLE = 0xffffc001L
    }
}

/** Bounded readers for Sony's v2/v3 datasets (protocol reference pinned in mobile/docs). */
internal object SonyPtpData {
    data class Extension(val version: Int, val codes: List<Int>)
    data class Property(val type: Int, val value: Long?)

    fun extension(bytes: ByteArray): Extension = read(bytes) { bb ->
        val version = bb.u16()
        if (version != 200 && version != 300) throw CameraSetupException("Sony returned an unsupported PC Remote protocol version: $version.")
        val codes = buildList {
            repeat(2) {
                val count = bb.count(2)
                repeat(count) { add(bb.u16()) }
            }
        }
        if (bb.hasRemaining()) throw PtpException("Unexpected Sony capability data.")
        Extension(version, codes)
    }

    fun properties(bytes: ByteArray): Map<Int, Property> = read(bytes) { bb ->
        val count = bb.count(7)
        bb.int // reserved
        val result = LinkedHashMap<Int, Property>()
        repeat(count) {
            val code = bb.u16()
            if (result.containsKey(code)) throw PtpException("Duplicate Sony property descriptor.")
            result[code] = descriptor(bb)
        }
        if (bb.hasRemaining()) throw PtpException("Unexpected Sony property data.")
        result
    }

    fun property(bytes: ByteArray, expectedCode: Int): Property = read(bytes) { bb ->
        if (bb.u16() != expectedCode) throw PtpException("Sony returned the wrong property.")
        descriptor(bb).also { if (bb.hasRemaining()) throw PtpException("Unexpected Sony property data.") }
    }

    private fun descriptor(bb: ByteBuffer): Property {
        val type = bb.u16()
        bb.get(); bb.get() // writable and enabled flags are irrelevant to read-only ingest
        bb.value(type) // default
        val current = bb.value(type)
        val form = bb.get().toInt() and 0xff
        when (form) {
            0 -> Unit
            1 -> repeat(3) { bb.value(type) }
            2 -> repeat(bb.u16()) { bb.value(type) }
            else -> throw PtpException("Unsupported Sony property form.")
        }
        // ponytail: upstream's <512 secondary-enum heuristic; use a versioned layout if a body exceeds it.
        // A following descriptor starts at 0x5xxx/0xdxxx.
        if (bb.remaining() >= 2 && (bb.getShort(bb.position()).toInt() and 0xffff) < 0x200) {
            val count = bb.u16()
            if (form != 2 && count != 0) throw PtpException("Unexpected Sony secondary property values.")
            repeat(count) { bb.value(type) }
        }
        return Property(type, current)
    }

    private fun ByteBuffer.value(type: Int): Long? {
        if (type == 0xffff) {
            val chars = get().toInt() and 0xff
            skip(chars * 2)
            return null
        }
        val base = type and 0x3fff
        val width = when (base) { 1, 2 -> 1; 3, 4 -> 2; 5, 6 -> 4; 7, 8 -> 8; 9, 10 -> 16
            else -> throw PtpException("Unsupported Sony property type.") }
        if ((type and 0x8000) != 0) throw PtpException("Unsupported Sony property type.")
        if ((type and 0x4000) != 0) {
            skip(count(width) * width)
            return null
        }
        return when (base) {
            1 -> get().toLong()
            2 -> (get().toInt() and 0xff).toLong()
            3 -> short.toLong()
            4 -> u16().toLong()
            5 -> int.toLong()
            6 -> int.toLong() and 0xffffffffL
            7, 8 -> long
            else -> { skip(width); null }
        }
    }

    private fun ByteBuffer.count(width: Int): Int {
        val count = int
        if (count < 0 || count > 100_000 || count > remaining() / width)
            throw PtpException("Invalid Sony dataset count.")
        return count
    }
    private fun ByteBuffer.skip(size: Int) {
        if (size > remaining()) throw PtpException("Truncated Sony property value.")
        position(position() + size)
    }
    private fun ByteBuffer.u16() = short.toInt() and 0xffff
    private fun <T> read(bytes: ByteArray, action: (ByteBuffer) -> T): T = try {
        action(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN))
    } catch (e: BufferUnderflowException) { throw PtpException("Truncated Sony dataset.", e) }
}
