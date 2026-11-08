package com.quickpitik.config

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.task.TaskExecutor
import org.springframework.scheduling.annotation.EnableAsync
import org.springframework.scheduling.annotation.EnableScheduling
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import java.time.Clock

@Configuration
@EnableAsync
@EnableScheduling
class AsyncConfig {

    @Bean
    fun clock(): Clock = Clock.systemUTC()

    // AI indexing: ~155 photos/min on 4 workers at race-day load. Six workers
    // keep downstream capacity ahead of verification slots to avoid queueing.
    @Bean(name = ["imageProcessing"])
    fun imageProcessingExecutor(): TaskExecutor = ThreadPoolTaskExecutor().apply {
        corePoolSize = 6
        maxPoolSize = corePoolSize
        // A race burst (1,000 frames in minutes) used to overflow 200 and fall
        // back to the 60 s sweep. Tasks are a photo id each — cheap to hold.
        queueCapacity = 2000
        setThreadNamePrefix("image-processing-")
        setWaitForTasksToCompleteOnShutdown(true)
        setAwaitTerminationSeconds(30)
        initialize()
    }

    // Watermarking peaked at 228 photos/min on race day. A separate pool keeps
    // AI calls from delaying previews going LIVE for runners.
    @Bean(name = ["watermarkProcessing"])
    fun watermarkProcessingExecutor(): ThreadPoolTaskExecutor = ThreadPoolTaskExecutor().apply {
        // Keep downstream capacity ahead of verification slots to avoid queueing.
        corePoolSize = 6
        maxPoolSize = corePoolSize
        queueCapacity = 2000
        setThreadNamePrefix("watermark-")
        setWaitForTasksToCompleteOnShutdown(true)
        setAwaitTerminationSeconds(30)
        initialize()
    }

    // Originals are reused after LIVE. Keep metadata off the watermark workers
    // with at most three 25 MiB originals retained; saturation recovers from DB.
    @Bean(name = ["metadataProcessing"])
    fun metadataProcessingExecutor(): ThreadPoolTaskExecutor = ThreadPoolTaskExecutor().apply {
        // ponytail: one worker/two queued originals; saturation uses fenced recovery.
        corePoolSize = 1
        maxPoolSize = 1
        queueCapacity = 2
        setThreadNamePrefix("metadata-")
        setWaitForTasksToCompleteOnShutdown(true)
        setAwaitTerminationSeconds(30)
        initialize()
    }

    // Email (Resend) gets its own pool for the same reason, one failure mode
    // further out: a Resend outage costs 3 attempts with sleeps per send, so on
    // the shared pool a race-finish burst of receipts could occupy every
    // indexing thread until it cleared (2026-09-11 architecture audit, P-4).
    // Smaller than the image pools — mail volume is per order, not per photo.
    @Bean(name = ["emailSending"])
    fun emailSendingExecutor(): TaskExecutor = ThreadPoolTaskExecutor().apply {
        corePoolSize = 2
        maxPoolSize = corePoolSize
        queueCapacity = 1000
        setThreadNamePrefix("email-")
        setWaitForTasksToCompleteOnShutdown(true)
        setAwaitTerminationSeconds(30)
        initialize()
    }
}
