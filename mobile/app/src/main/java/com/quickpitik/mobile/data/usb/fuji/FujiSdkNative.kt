package com.quickpitik.mobile.data.usb.fuji

/** JNI binding over src/main/cpp/fuji_sdk.cpp. Only loadable in builds with the SDK (BuildConfig.FUJI_SDK). */
object FujiSdkNative : FujiSdkApi {
    private external fun nativeLoad(): Long
    private external fun nativeInit(): Long
    private external fun nativeExit(): Long
    private external fun nativeOpen(fd: Int, raw: ByteArray, out: LongArray): Long
    private external fun nativeClose(handle: Long): Long
    private external fun nativeError(handle: Long, out: LongArray): Long
    private external fun nativeDeviceInfo(handle: Long): Array<String>?
    private external fun nativeGetLong(handle: Long, what: Int, out: LongArray): Long
    private external fun nativeSetMediaRecord(handle: Long, mode: Long): Long
    private external fun nativeReadImageInfo(handle: Long, out: LongArray, name: Array<String?>): Long
    private external fun nativeReadImage(handle: Long, into: ByteArray): Long
    private external fun nativeDeleteImage(handle: Long): Long

    private fun check(handle: Long, result: Long) {
        if (result == 0L) return
        val out = LongArray(2)
        nativeError(handle, out)
        throw FujiSdkException(out[0], out[1])
    }

    override fun load() {
        System.loadLibrary("fujisdk")
        if (nativeLoad() != 0L) throw FujiSdkException(0, 0x1004) // XSDK_ERRCODE_LOADLIB
    }
    override fun init() = check(0, nativeInit())
    override fun exit() { nativeExit() }
    override fun open(fileDescriptor: Int, rawDescriptors: ByteArray): Long {
        val out = LongArray(1)
        check(0, nativeOpen(fileDescriptor, rawDescriptors, out))
        return out[0]
    }
    override fun close(handle: Long) { nativeClose(handle) }
    override fun deviceInfo(handle: Long): FujiDeviceInfo {
        val f = nativeDeviceInfo(handle) ?: throw FujiSdkException(0x1041, 0x2001)
        return FujiDeviceInfo(f[0], f[1], f[2], f[3])
    }
    override fun mode(handle: Long) = LongArray(1).also { check(handle, nativeGetLong(handle, 0, it)) }[0]
    override fun mediaRecord(handle: Long) = LongArray(1).also { check(handle, nativeGetLong(handle, 1, it)) }[0]
    override fun setMediaRecord(handle: Long, mode: Long) = check(handle, nativeSetMediaRecord(handle, mode))
    override fun readImageInfo(handle: Long): FujiImageInfo {
        val out = LongArray(2)
        val name = arrayOfNulls<String>(1)
        try {
            check(handle, nativeReadImageInfo(handle, out, name))
        } catch (e: FujiSdkException) {
            // Fujifilm's own sample retries on exactly these three; treat them as "nothing yet".
            if (e.code in setOf(FujiSdk.ERR_BUSY, FujiSdk.ERR_FORCEMODE_BUSY, FujiSdk.ERR_RUNNING_OTHER_FUNCTION))
                return FujiImageInfo("", FujiSdk.FORMAT_NONE.toLong(), 0)
            throw e
        }
        return FujiImageInfo(name[0].orEmpty(), out[0], out[1])
    }
    override fun readImage(handle: Long, into: ByteArray): Int { check(handle, nativeReadImage(handle, into)); return into.size }
    override fun deleteImage(handle: Long) = check(handle, nativeDeleteImage(handle))
}
