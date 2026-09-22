package com.quickpitik.mobile.data.usb.fuji

import com.quickpitik.mobile.data.usb.ptp.CameraCapabilities
import com.quickpitik.mobile.data.usb.ptp.CameraSetupException
import com.quickpitik.mobile.data.usb.ptp.LiveCamera
import com.quickpitik.mobile.data.usb.ptp.Ptp
import com.quickpitik.mobile.data.usb.ptp.PtpDeviceInfo
import com.quickpitik.mobile.data.usb.ptp.PtpObjectInfo
import java.io.OutputStream

/**
 * Fujifilm Camera Priority tethering through the official SDK: the photographer shoots with
 * the body, each frame lands in the in-camera buffer, and we pull JPEGs from the top of it.
 * Card import is not possible through the SDK (storage model, SDK reference §1.5); it stays on PTP.
 */
class FujiSdkCamera(
    private val api: FujiSdkApi,
    private val handle: Long,
    override val info: PtpDeviceInfo,
    override val cameraKey: String,
    private val release: () -> Unit,
) : LiveCamera {
    private var closed = false
    override val setupWarning: String? = null
    override val isUsable: Boolean get() = !closed
    override val capabilities = CameraCapabilities(
        cardImport = false, thumbnails = false, eventPolling = false,
        physicalShutter = "Fujifilm SDK, Camera Priority: RAW+JPEG card recording and an empty buffer are required before auto-upload starts.",
        imageFormats = listOf(Ptp.OBJECT_FORMAT_EXIF_JPEG), transientObjects = true,
        assumptions = listOf(
            "Camera Priority should keep the body's controls and shutter (SDK reference §1.7); hardware verification is still required.",
            "Only recognized RAW or HEIF companions are discarded, and only while verified RAW+JPEG card recording remains active.",
        ),
    )

    /** Verify a safe new session before any buffered frame can be consumed. Called by the opener. */
    fun start() {
        if (api.mode(handle) >= FujiSdk.MODE_MOVIE_FIRST)
            throw CameraSetupException("Switch the camera to a still-photo mode (P, A, S or M), then start again.")
        requireSafeRecordingMode()
        val pending = try {
            api.readImageInfo(handle)
        } catch (e: FujiSdkException) {
            throw CameraSetupException("The Fujifilm buffer could not be verified empty. Save existing buffered shots with the camera/vendor workflow, then start again.")
        }
        if (pending.baseFormat != FujiSdk.FORMAT_NONE)
            throw CameraSetupException("The Fujifilm buffer already contains a shot. Save it with the camera/vendor workflow before starting; changing settings now does not back it up.")
    }

    override fun nextCapture(): Pair<Long, PtpObjectInfo>? {
        while (true) {
            requireSafeRecordingMode()
            val image = try {
                api.readImageInfo(handle)
            } catch (e: FujiSdkException) {
                if (e.code in BUSY_ERRORS) return null
                throw e
            }
            when (image.baseFormat) {
                FujiSdk.FORMAT_NONE -> return null
                FujiSdk.FORMAT_JPEG -> return FujiSdk.BUFFER_HANDLE to PtpObjectInfo(
                    Ptp.OBJECT_FORMAT_EXIF_JPEG, image.size,
                    image.name.ifBlank { "FUJI_%d.JPG".format(System.currentTimeMillis()) },
                )
                FujiSdk.FORMAT_RAW, FujiSdk.FORMAT_HEIF -> api.deleteImage(handle)
                else -> throw CameraSetupException("The Fujifilm buffer contains an unknown format. Preserve it with the camera/vendor workflow before restarting auto-upload.")
            }
        }
    }

    override fun readObject(handle: Long, sink: OutputStream, expectedSize: Long?): Long {
        if (handle != FujiSdk.BUFFER_HANDLE) throw CameraSetupException("The Fujifilm buffer changed. Restart auto-upload after preserving its contents.")
        requireSafeRecordingMode()
        val image = try {
            api.readImageInfo(this.handle)
        } catch (e: FujiSdkException) {
            throw CameraSetupException("The Fujifilm buffer could not be verified before transfer. Preserve it with the camera/vendor workflow, then restart auto-upload.")
        }
        if (image.baseFormat != FujiSdk.FORMAT_JPEG)
            throw CameraSetupException("The Fujifilm buffer no longer contains the expected JPEG. Preserve it with the camera/vendor workflow, then restart auto-upload.")
        val size = requireNotNull(expectedSize) { "Fujifilm frames always carry a size." }.toInt()
        if (image.size != expectedSize)
            throw CameraSetupException("The Fujifilm buffer changed before transfer. Preserve it with the camera/vendor workflow, then restart auto-upload.")
        val bytes = ByteArray(size)
        val read = api.readImage(this.handle, bytes)
        sink.write(bytes, 0, read)
        return read.toLong()
    }

    override fun handles(): List<Long> = emptyList()
    override fun events(): List<Long> = emptyList()
    override fun objectInfo(handle: Long): PtpObjectInfo =
        throw CameraSetupException("Card files are not readable through the Fujifilm SDK. Use USB CARD READER mode for import.")
    override fun keepAlive() {}

    private fun requireSafeRecordingMode() {
        val recording = try {
            api.mediaRecord(handle)
        } catch (e: FujiSdkException) {
            throw CameraSetupException("The Fujifilm card recording mode could not be verified. Preserve buffered shots and restart auto-upload.")
        }
        if (recording != FujiSdk.MEDIAREC_RAWJPEG)
            throw CameraSetupException("Set Fujifilm tether card recording to RAW+JPEG, save any existing buffered shots with the camera/vendor workflow, then start again.")
    }

    override fun close() {
        if (closed) return
        closed = true
        runCatching { api.close(handle) }
        runCatching { api.exit() }
        release()
    }

    private companion object {
        val BUSY_ERRORS = setOf(FujiSdk.ERR_BUSY, FujiSdk.ERR_FORCEMODE_BUSY, FujiSdk.ERR_RUNNING_OTHER_FUNCTION)
    }
}
