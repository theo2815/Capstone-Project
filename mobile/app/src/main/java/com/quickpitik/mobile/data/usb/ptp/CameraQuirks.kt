package com.quickpitik.mobile.data.usb.ptp

/**
 * Demonstrated exceptions only. A body that matches no rule runs on standard PTP plus the
 * adaptive checks in the drivers and the live loop, so adding a model is one row here.
 * Sources: the R6/R10 hardware journals and libgphoto2's ptp2 camlib at the pinned commit.
 */
data class CameraQuirks(
    /** Canon EOS remote-mode value; null sends no SetRemoteMode at all (PowerShot SX600 HS in libgphoto2). */
    val eosRemoteMode: Int? = 1,
    /** False where vendor event polling is known to be useless (R6/R10: 0 of 111 body shots) or harmful (Nikon 1). */
    val vendorEvents: Boolean = true,
    /** Throwaway standard session before the first vendor op after attach (EOS R10 first-session stall). */
    val warmUp: Boolean = false,
    /** libgphoto2 PTP_DONT_CLOSE_SESSION: these 2016-era EOS bodies answer 0x2005 to everything after CloseSession. */
    val skipCloseSession: Boolean = false,
    /** Shown in the capture log so the photographer sees why this body is treated specially. */
    val note: String? = null,
) {
    companion object {
        private class Rule(
            val vid: Int? = null,
            val pid: Int? = null,
            val model: Regex? = null,
            /** The rule applies only when the camera advertises this operation (libgphoto2 is_canon_eos_m). */
            val requiresOp: Int? = null,
            val quirks: CameraQuirks,
        )

        private const val R10_NOTE = "EOS R10: warm-up session before the first vendor op after attach; EOS events reported no body shots (2026-09-04)."
        private const val EOS_M_NOTE = "EOS M / PowerShot family: remote mode 0x15 per libgphoto2; body-shutter behaviour untested here."
        private const val NO_CLOSE_NOTE = "CloseSession is skipped at the end of a session: libgphoto2 reports this body rejects every command after it."

        // First match wins: specific bodies before families.
        private val rules = listOf(
            Rule(vid = 0x04a9, pid = 0x32f8, quirks = CameraQuirks(vendorEvents = false, warmUp = true, note = R10_NOTE)),
            Rule(model = Regex("(?i)^(Canon )?EOS R10$"), quirks = CameraQuirks(vendorEvents = false, warmUp = true, note = R10_NOTE)),
            Rule(model = Regex("(?i)^(Canon )?EOS R6$"), quirks = CameraQuirks(vendorEvents = false,
                note = "EOS R6: EOS events reported 0 of 111 body shots; the card scan is the detector (2026-09-02).")),
            Rule(model = Regex("(?i)^Canon EOS (5D Mark IV|80D|5DS|5DS R|R|RP|2000D|1500D|Rebel T7|Kiss X90)$"),
                quirks = CameraQuirks(skipCloseSession = true, note = NO_CLOSE_NOTE)),
            Rule(model = Regex("(?i)^Canon EOS M6 Mark II$"), quirks = CameraQuirks(note = "EOS M6 Mark II behaves like a regular EOS (libgphoto2).")),
            Rule(model = Regex("(?i)^Canon EOS M"), quirks = CameraQuirks(eosRemoteMode = 0x15, note = EOS_M_NOTE)),
            Rule(model = Regex("(?i)^Canon PowerShot SX600 HS$"), requiresOp = Ptp.OP_EOS_REMOTE_RELEASE_ON,
                quirks = CameraQuirks(eosRemoteMode = null, note = "PowerShot SX600 HS: libgphoto2 sends no remote mode on this body.")),
            Rule(model = Regex("(?i)^Canon PowerShot (SX720 HS|SX540 HS|G5 X)$"), requiresOp = Ptp.OP_EOS_REMOTE_RELEASE_ON,
                quirks = CameraQuirks(eosRemoteMode = 0x11, note = "PowerShot SX720 HS / SX540 HS / G5 X: remote mode 0x11 (libgphoto2).")),
            Rule(model = Regex("(?i)^Canon (PowerShot (SX|G)|Digital IXUS)"), requiresOp = Ptp.OP_EOS_REMOTE_RELEASE_ON,
                quirks = CameraQuirks(eosRemoteMode = 0x15, note = EOS_M_NOTE)),
            Rule(model = Regex("""(?i)\bNikon\s+1\b"""), quirks = CameraQuirks(vendorEvents = false,
                note = "Nikon 1: vendor event commands break the session (libgphoto2); card scan only.")),
        )

        fun forCamera(vid: Int, pid: Int, model: String, operations: Collection<Int> = emptyList()): CameraQuirks =
            rules.firstOrNull { rule ->
                (rule.vid == null || rule.vid == vid) && (rule.pid == null || rule.pid == pid) &&
                    (rule.model == null || rule.model.containsMatchIn(model)) &&
                    (rule.requiresOp == null || rule.requiresOp in operations)
            }?.quirks ?: CameraQuirks()
    }
}
