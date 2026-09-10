package com.quickpitik.mobile.data.usb.ptp

/**
 * Demonstrated exceptions only. A body that matches no rule runs on standard PTP plus the
 * adaptive checks in the drivers and the live loop, so adding a model is one row here.
 */
data class CameraQuirks(
    /** Canon EOS remote-mode value. libgphoto2 uses 0x15 on EOS M / newer PowerShot SX, 0x11 on SX720/SX600, else 1. */
    val eosRemoteMode: Int = 1,
    /** False where vendor event polling is known to be useless (R6/R10: 0 of 111 body shots) or harmful (Nikon 1). */
    val vendorEvents: Boolean = true,
    /** Throwaway standard session before the first vendor op after attach (EOS R10 first-session stall). */
    val warmUp: Boolean = false,
    /** Shown in the capture log so the photographer sees why this body is treated specially. */
    val note: String? = null,
) {
    companion object {
        private class Rule(val vid: Int? = null, val pid: Int? = null, val model: Regex? = null, val quirks: CameraQuirks)

        private const val R10_NOTE = "EOS R10: warm-up session before the first vendor op after attach; EOS events reported no body shots (2026-09-04)."

        // First match wins: specific bodies before families.
        private val rules = listOf(
            Rule(vid = 0x04a9, pid = 0x32f8, quirks = CameraQuirks(vendorEvents = false, warmUp = true, note = R10_NOTE)),
            Rule(model = Regex("(?i)^(Canon )?EOS R10$"), quirks = CameraQuirks(vendorEvents = false, warmUp = true, note = R10_NOTE)),
            Rule(model = Regex("(?i)^(Canon )?EOS R6$"), quirks = CameraQuirks(vendorEvents = false,
                note = "EOS R6: EOS events reported 0 of 111 body shots; the card scan is the detector (2026-09-02).")),
            Rule(model = Regex("(?i)^Canon EOS M6 Mark II$"), quirks = CameraQuirks(note = "EOS M6 Mark II behaves like a regular EOS (libgphoto2).")),
            Rule(model = Regex("(?i)^Canon PowerShot SX(720|600) HS$"), quirks = CameraQuirks(eosRemoteMode = 0x11,
                note = "PowerShot SX720/SX600: remote mode 0x11 (libgphoto2).")),
            Rule(model = Regex("(?i)^Canon (EOS M|PowerShot SX)"), quirks = CameraQuirks(eosRemoteMode = 0x15,
                note = "EOS M / PowerShot SX: remote mode 0x15 (libgphoto2); untested here.")),
            Rule(model = Regex("""(?i)\bNikon\s+1\b"""), quirks = CameraQuirks(vendorEvents = false,
                note = "Nikon 1: vendor event commands break the session (libgphoto2); card scan only.")),
        )

        fun forCamera(vid: Int, pid: Int, model: String): CameraQuirks = rules.firstOrNull { rule ->
            (rule.vid == null || rule.vid == vid) && (rule.pid == null || rule.pid == pid) &&
                (rule.model == null || rule.model.containsMatchIn(model))
        }?.quirks ?: CameraQuirks()
    }
}
