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
    override var setupWarning: String? = null
    override val isUsable: Boolean get() = !closed
    override val capabilities = CameraCapabilities(
        cardImport = false, thumbnails = false, eventPolling = false,
        physicalShutter = "Fujifilm SDK, Camera Priority: shoot with the body; each JPEG is pulled from the camera buffer.",
        imageFormats = listOf(Ptp.OBJECT_FORMAT_EXIF_JPEG), transientObjects = true,
        assumptions = listOf(
            "Camera Priority is the SDK default and keeps the body's controls and shutter (SDK reference §1.7); verified only on the body used in the spike.",
            "RAW and HEIF frames are removed from the buffer without upload; their card copy depends on the camera's media-record setting, which is forced to RAW+JPEG only when it was OFF.",
        ),
    )

    /** Still mode and a card copy before the first frame. Called by the opener. */
    fun start() {
        if (api.mode(handle) >= FujiSdk.MODE_MOVIE_FIRST)
            throw CameraSetupException("Switch the camera to a still-photo mode (P, A, S or M), then start again.")
        when (api.mediaRecord(handle)) {
            FujiSdk.MEDIAREC_OFF -> {
                api.setMediaRecord(handle, FujiSdk.MEDIAREC_RAWJPEG)
                setupWarning = "The camera was not saving tethered shots to its card; RAW+JPEG card recording was turned on so every shot keeps a backup."
            }
            FujiSdk.MEDIAREC_JPEG, FujiSdk.MEDIAREC_RAW ->
                setupWarning = "The camera saves only one format to its card while tethered; frames of the other format exist only on the phone."
        }
    }

    override fun nextCapture(): Pair<Long, PtpObjectInfo>? {
        while (true) {
            val image = api.readImageInfo(handle)
            when (image.baseFormat) {
                FujiSdk.FORMAT_NONE -> return null
                FujiSdk.FORMAT_JPEG -> return FujiSdk.BUFFER_HANDLE to PtpObjectInfo(
                    Ptp.OBJECT_FORMAT_EXIF_JPEG, image.size,
                    image.name.ifBlank { "FUJI_%d.JPG".format(System.currentTimeMillis()) },
                )
                else -> api.deleteImage(handle) // RAW / HEIF: not uploadable; the card copy is the photographer's
            }
        }
    }

    override fun readObject(handle: Long, sink: OutputStream, expectedSize: Long?): Long {
        val size = requireNotNull(expectedSize) { "Fujifilm frames always carry a size." }.toInt()
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

    override fun close() {
        if (closed) return
        closed = true
        runCatching { api.close(handle) }
        runCatching { api.exit() }
        release()
    }
}
