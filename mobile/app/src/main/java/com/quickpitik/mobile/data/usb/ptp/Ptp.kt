package com.quickpitik.mobile.data.usb.ptp

/**
 * PTP (Picture Transfer Protocol, ISO 15740) constants used by the tether layer.
 *
 * Only the codes we actually reference live here; the probe dumps every code the
 * camera reports as raw hex, so we don't need an exhaustive table to read a new
 * body's capabilities. Canon EOS bodies (like the R6) layer a vendor extension
 * on top of standard PTP — its operations sit in the 0x9xxx range and capture
 * events arrive through a polled EOS_GetEvent loop, not the standard interrupt
 * ObjectAdded event. We confirm which path a body supports from GetDeviceInfo.
 */
object Ptp {
    // USB container types (first field after the length header).
    const val TYPE_COMMAND = 1
    const val TYPE_DATA = 2
    const val TYPE_RESPONSE = 3
    const val TYPE_EVENT = 4

    // Standard operations.
    const val OP_GET_DEVICE_INFO = 0x1001
    const val OP_OPEN_SESSION = 0x1002
    const val OP_CLOSE_SESSION = 0x1003
    const val OP_GET_STORAGE_IDS = 0x1004
    const val OP_GET_OBJECT_HANDLES = 0x1007
    const val OP_GET_OBJECT_INFO = 0x1008
    const val OP_GET_OBJECT = 0x1009
    const val OP_INITIATE_CAPTURE = 0x100E

    // Standard event: a new object (photo) appeared on the device. Many cameras
    // push this on the interrupt endpoint; Canon EOS instead surfaces captures
    // inside the EOS_GetEvent payload.
    const val EV_OBJECT_ADDED = 0x4002

    // ObjectInfo.ObjectFormat codes (standard PTP). The card-poll path enumerates
    // the card with GetObjectHandles, then reads each ObjectInfo to decide what to
    // pull: only the JPEG, never folders or RAW. (RAW stays on the card — the
    // photographer keeps shooting RAW for their deliverables; the JPEG goes up for
    // instant search/sale.)
    const val OBJECT_FORMAT_ASSOCIATION = 0x3001 // a folder / directory, not a file
    const val OBJECT_FORMAT_EXIF_JPEG = 0x3801   // the camera-written JPEG we want

    // Canon EOS vendor-extension operations (from libgphoto2's ptp2 camlib;
    // confirmed present on the EOS R6 by the on-device probe). Presence of
    // EOS_GET_EVENT in OperationsSupported is our EOS-body signal.
    const val OP_EOS_GET_OBJECT = 0x9104
    const val OP_EOS_GET_PARTIAL_OBJECT = 0x9107
    const val OP_EOS_SET_DEVICE_PROP_VALUE_EX = 0x9110
    const val OP_EOS_SET_REMOTE_MODE = 0x9114
    const val OP_EOS_SET_EVENT_MODE = 0x9115
    const val OP_EOS_GET_EVENT = 0x9116
    const val OP_EOS_TRANSFER_COMPLETE = 0x9117
    const val OP_EOS_KEEP_DEVICE_ON = 0x911D
    // Host-triggered shutter. Canon reports host-initiated captures (unlike a
    // local physical-shutter press in remote mode); RemoteReleaseOn/Off is the
    // press-and-release sequence on modern EOS bodies (R6 advertises both).
    const val OP_EOS_REMOTE_RELEASE = 0x910F
    const val OP_EOS_REMOTE_RELEASE_ON = 0x9128
    const val OP_EOS_REMOTE_RELEASE_OFF = 0x9129

    // Canon EOS device property: where a capture is stored. Setting it makes the
    // body actually report/transfer host-triggered shots. 4 = transfer to host.
    const val EOS_DPC_CAPTURE_DESTINATION = 0xD11C
    const val EOS_CAPTUREDEST_HOST = 4

    // Canon EOS events (delivered inside the EOS_GetEvent payload, NOT on the
    // standard interrupt endpoint). These signal a freshly captured frame.
    const val EOS_EVENT_OBJECT_ADDED_EX = 0xC181
    const val EOS_EVENT_REQUEST_OBJECT_TRANSFER = 0xC186
    const val EOS_EVENT_OBJECT_ADDED_EX64 = 0xC191

    // Response codes.
    const val RC_OK = 0x2001
    const val RC_SESSION_ALREADY_OPEN = 0x201E

    // PTP VendorExtensionID values (DeviceInfo).
    private const val VENDOR_MICROSOFT = 0x06L
    private const val VENDOR_NIKON = 0x0AL
    private const val VENDOR_CANON = 0x0BL
    private const val VENDOR_SONY = 0x11L

    fun vendorName(id: Long): String = when (id) {
        VENDOR_MICROSOFT -> "Microsoft/MTP"
        VENDOR_NIKON -> "Nikon"
        VENDOR_CANON -> "Canon"
        VENDOR_SONY -> "Sony"
        0L -> "none"
        else -> "unknown"
    }
}
