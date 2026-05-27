package com.quickpitik.mobile.data.usb.ptp

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Parsed PTP DeviceInfo dataset (the payload of the GetDeviceInfo data phase).
 * This is the camera's self-description: which protocol extension it speaks and
 * which operations/events it supports — exactly what we need to decide how to
 * pull captures off a given body.
 */
data class PtpDeviceInfo(
    val standardVersion: Int,
    val vendorExtensionId: Long,
    val vendorExtensionVersion: Int,
    val vendorExtensionDesc: String,
    val operationsSupported: List<Int>,
    val eventsSupported: List<Int>,
    val devicePropertiesSupported: List<Int>,
    val captureFormats: List<Int>,
    val imageFormats: List<Int>,
    val manufacturer: String,
    val model: String,
    val deviceVersion: String,
    val serialNumber: String,
) {
    companion object {
        /** Parse the DeviceInfo dataset (bytes AFTER the 12-byte container header). */
        fun parse(payload: ByteArray): PtpDeviceInfo {
            val buf = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN)
            val standardVersion = buf.u16()
            val vendorExtensionId = buf.u32()
            val vendorExtensionVersion = buf.u16()
            val vendorExtensionDesc = buf.ptpString()
            buf.u16() // FunctionalMode — unused
            val operations = buf.u16Array()
            val events = buf.u16Array()
            val deviceProps = buf.u16Array()
            val captureFormats = buf.u16Array()
            val imageFormats = buf.u16Array()
            val manufacturer = buf.ptpString()
            val model = buf.ptpString()
            val deviceVersion = buf.ptpString()
            val serialNumber = buf.ptpString()
            return PtpDeviceInfo(
                standardVersion = standardVersion,
                vendorExtensionId = vendorExtensionId,
                vendorExtensionVersion = vendorExtensionVersion,
                vendorExtensionDesc = vendorExtensionDesc,
                operationsSupported = operations,
                eventsSupported = events,
                devicePropertiesSupported = deviceProps,
                captureFormats = captureFormats,
                imageFormats = imageFormats,
                manufacturer = manufacturer,
                model = model,
                deviceVersion = deviceVersion,
                serialNumber = serialNumber,
            )
        }

        private fun ByteBuffer.u16(): Int = short.toInt() and 0xFFFF
        private fun ByteBuffer.u32(): Long = int.toLong() and 0xFFFFFFFFL

        // PTP UINT16 array: UINT32 count, then count UINT16 elements.
        private fun ByteBuffer.u16Array(): List<Int> {
            val count = int
            if (count <= 0) return emptyList()
            return ArrayList<Int>(count).apply { repeat(count) { add(u16()) } }
        }

        // PTP string: UINT8 length (chars incl. trailing NUL), then UTF-16LE.
        // We append every code unit except the NUL terminator (code 0), so
        // legitimate spaces inside a model name like "Canon EOS R6" survive.
        private fun ByteBuffer.ptpString(): String {
            val len = get().toInt() and 0xFF
            if (len == 0) return ""
            val sb = StringBuilder(len)
            for (i in 0 until len) {
                val code = short.toInt() and 0xFFFF
                if (code != 0) sb.append(code.toChar())
            }
            return sb.toString().trim()
        }
    }
}
