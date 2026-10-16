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

    // AI indexing (Rekognition / ai-api): ~9 s of pool time per photo.
    @Bean(name = ["imageProcessing"])
    fun imageProcessingExecutor(): TaskExecutor = ThreadPoolTaskExecutor().apply {
        corePoolSize = 4
        maxPoolSize = corePoolSize
        // A race burst (1,000 frames in minutes) used to overflow 200 and fall
        // back to the 60 s sweep. Tasks are a photo id each — cheap to hold.
        queueCapacity = 2000
        setThreadNamePrefix("image-processing-")
        setWaitForTasksToCompleteOnShutdown(true)
        setAwaitTerminationSeconds(30)
        initialize()
    }

    // Watermarking (storage GET + composite + PUT, ~3-5 s) gets its own pool so
    // the slow AI calls above can't sit between an upload and the photo going
    // LIVE for runners (2026-09-02 scale review).
    @Bean(name = ["watermarkProcessing"])
    fun watermarkProcessingExecutor(): ThreadPoolTaskExecutor = ThreadPoolTaskExecutor().apply {
        // 2 until SubsampledDecode (P0-6) + the 2 GB limit (OPS-2) land, then back to 4
        corePoolSize = 2
        maxPoolSize = 2
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
