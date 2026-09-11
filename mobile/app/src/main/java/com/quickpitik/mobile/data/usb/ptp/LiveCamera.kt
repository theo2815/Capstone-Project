package com.quickpitik.mobile.data.usb.ptp

import java.io.Closeable
import java.io.OutputStream

/** What the live loop needs from a camera. PTP bodies and the Fujifilm SDK both provide it. */
interface LiveCamera : Closeable {
    val info: PtpDeviceInfo
    val cameraKey: String
    val capabilities: CameraCapabilities
    val setupWarning: String?
    /** False once the transport is broken; the loop reopens instead of retrying on it. */
    val isUsable: Boolean
    fun handles(): List<Long>
    fun events(): List<Long>
    fun objectInfo(handle: Long): PtpObjectInfo
    fun readObject(handle: Long, sink: OutputStream, expectedSize: Long?): Long
    fun nextCapture(): Pair<Long, PtpObjectInfo>?
    fun keepAlive()
}
