package com.quickpitik.mobile.data.usb.fuji

import com.quickpitik.mobile.data.usb.ptp.PtpException

/** The dozen SDK calls the tether path needs. The native binding is FujiSdkNative; tests use a fake. */
interface FujiSdkApi {
    fun load()
    fun init()
    fun exit()
    /** Opens the camera over an Android USB file descriptor; returns the SDK camera handle. */
    fun open(fileDescriptor: Int, rawDescriptors: ByteArray): Long
    fun close(handle: Long)
    fun deviceInfo(handle: Long): FujiDeviceInfo
    fun mode(handle: Long): Long
    fun mediaRecord(handle: Long): Long
    fun setMediaRecord(handle: Long, mode: Long)
    /** Top of the in-camera buffer; format NONE when it is empty or the camera is momentarily busy. */
    fun readImageInfo(handle: Long): FujiImageInfo
    /** Reads and removes the top image; returns the bytes written into [into]. */
    fun readImage(handle: Long, into: ByteArray): Int
    fun deleteImage(handle: Long)
}

data class FujiDeviceInfo(val vendor: String, val product: String, val serial: String, val firmware: String)
data class FujiImageInfo(val name: String, val format: Long, val size: Long) {
    /** Low five bits are the format; the rest is rotation (XSDK_IMAGEFORMAT_*_90/180/270). */
    val baseFormat: Int get() = (format and 0x1F).toInt()
}
class FujiSdkException(val api: Long, val code: Long) : PtpException("Fujifilm SDK error 0x%X in API 0x%X".format(code, api))

object FujiSdk {
    const val FORMAT_RAW = 1
    const val FORMAT_NONE = 5
    const val FORMAT_JPEG = 7
    const val FORMAT_HEIF = 0x12
    const val MEDIAREC_RAWJPEG = 0x0001L
    const val MEDIAREC_RAW = 0x0002L
    const val MEDIAREC_JPEG = 0x0003L
    const val MEDIAREC_OFF = 0x0004L
    const val MODE_MOVIE_FIRST = 0x0100L
    const val ERR_BUSY = 0x1006L
    const val ERR_FRAME_FULL = 0x1009L
    const val ERR_NO_MODEL_MODULE = 0x1012L
    const val ERR_INVALID_USBMODE = 0x1015L
    const val ERR_FORCEMODE_BUSY = 0x1016L
    const val ERR_RUNNING_OTHER_FUNCTION = 0x1017L
    /** Sequential frames come out of one buffer; the handle is a label, as for Sony. */
    const val BUFFER_HANDLE = 0xfffff001L
    /** SDK reference 1.34 §1.2. Bodies outside this set still get the plain PTP path. */
    val supportedProducts = setOf(
        "X-T3", "X-T4", "X-Pro3", "X-S10", "X-H2S", "X-H2", "X-T5", "X-S20", "X-M5",
        "GFX 50S", "GFX 50R", "GFX 100", "GFX100S", "GFX50S II", "GFX100 II", "GFX100S II", "GFX100RF",
    )
}
