package com.quickpitik.service.relevance

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component

// The one place that turns the relevance settings into what may run now (plan section 5.2).
@Component
class RelevanceSwitch(
    relevance: RelevanceProperties,
    admin: AdminProperties,
    aiApi: AiApiProperties,
) {
    val cutoffsSet: Boolean = relevance.junk.blurCutoff != null && relevance.junk.darkCutoff != null
    val aiEnabled: Boolean = aiApi.enabled

    // Version 1's flag mode until RV-1.10 retires it (P2). With the admin flags switch off nobody could
    // see or close a flag, so it raises none.
    val v1Flagging: Boolean = relevance.mode == RelevanceMode.FLAG && admin.flagsEnabled

    // Always OFF, RECORD_ONLY or AUTO_HIDE.
    val mode: RelevanceMode = when (relevance.mode) {
        RelevanceMode.OFF -> RelevanceMode.OFF
        RelevanceMode.RECORD_ONLY, RelevanceMode.SHADOW, RelevanceMode.FLAG -> RelevanceMode.RECORD_ONLY
        RelevanceMode.AUTO_HIDE ->
            if (admin.flagsEnabled && cutoffsSet) RelevanceMode.AUTO_HIDE else RelevanceMode.RECORD_ONLY
    }

    init {
        val log = LoggerFactory.getLogger(javaClass)
        if (relevance.mode == RelevanceMode.FLAG && !admin.flagsEnabled) {
            log.warn("RELEVANCE_MODE=flag with ADMIN_FLAGS_ENABLED off: relevance runs as shadow, no flag is raised")
        }
        if (relevance.mode == RelevanceMode.AUTO_HIDE && mode != RelevanceMode.AUTO_HIDE) {
            val missing = listOfNotNull(
                "ADMIN_FLAGS_ENABLED".takeUnless { admin.flagsEnabled },
                "RELEVANCE_JUNK_BLUR_CUTOFF".takeIf { relevance.junk.blurCutoff == null },
                "RELEVANCE_JUNK_DARK_CUTOFF".takeIf { relevance.junk.darkCutoff == null },
            )
            log.warn("RELEVANCE_MODE=auto-hide runs as record-only, missing: {}", missing.joinToString())
        }
    }
}
