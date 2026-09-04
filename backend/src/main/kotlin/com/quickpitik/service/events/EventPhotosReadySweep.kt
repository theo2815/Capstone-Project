package com.quickpitik.service.events

import com.quickpitik.config.AiApiProperties
import com.quickpitik.repository.EventPhotoAlertRepository
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import java.time.LocalDate
import java.time.ZoneId

// Date-based sweep for the "your photos are ready" email. Events are
// day-granular (Event.date has no clock time), and photographers upload across
// the [date, date+3d] window, so this checks through date+4 (one final day for
// accepted uploads to finish indexing) and emails each opted-in runner once.
//
// Mirrors PhotoIndexingTrigger.reconcile — synchronous per-tick loop, gated on
// aiApiProperties.enabled, bounded batch. fixedDelay waits for a tick to finish
// before starting the next, so a slow provider can't overlap runs. Single-instance
// assumption like the other sweeps; the claimNotify conditional UPDATE keeps the
// send single even if two instances ever sweep at once.
@Component
class EventPhotosReadySweep(
    private val alertRepository: EventPhotoAlertRepository,
    private val notifier: EventPhotosReadyNotifier,
    private val aiApiProperties: AiApiProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Scheduled(fixedDelayString = "\${app.ai-api.photos-ready-sweep-interval-ms:3600000}")
    fun sweep() {
        if (!aiApiProperties.enabled) return // no AI = no possible match; no-op like the other sweeps
        val today = LocalDate.now(PH_ZONE)
        val pending = alertRepository.findPendingInWindow(
            today = today,
            windowStart = today.minusDays(UPLOAD_WINDOW_DAYS + POST_WINDOW_CHECK_DAYS),
            pageable = PageRequest.of(0, BATCH),
        )
        if (pending.isEmpty()) return
        log.info("Photos-ready sweep: {} pending opt-in(s)", pending.size)
        pending.forEach { alert ->
            try {
                notifier.notifyIfMatched(alert.id)
            } catch (ex: Exception) {
                log.warn("Photos-ready notify failed for alert {}: {}", alert.id, ex.message)
            }
        }
    }

    private companion object {
        val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        const val UPLOAD_WINDOW_DAYS = 3L // matches PhotoUploadService [date, date+3d]
        const val POST_WINDOW_CHECK_DAYS = 1L
        // ponytail: synchronous per-tick loop bounded at 100; dispatch to the
        // imageProcessing pool only if one event ever exceeds a few hundred opt-ins.
        const val BATCH = 100
    }
}
