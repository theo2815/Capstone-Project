package com.quickpitik.config

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.core.task.TaskExecutor
import org.springframework.scheduling.annotation.EnableAsync
import org.springframework.scheduling.annotation.EnableScheduling
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor

@Configuration
@EnableAsync
@EnableScheduling
class AsyncConfig {

    // AI indexing (Rekognition / ai-api): ~9 s of pool time per photo.
    @Bean(name = ["imageProcessing"])
    fun imageProcessingExecutor(): TaskExecutor = ThreadPoolTaskExecutor().apply {
        corePoolSize = 4
        maxPoolSize = 8
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
    fun watermarkProcessingExecutor(): TaskExecutor = ThreadPoolTaskExecutor().apply {
        corePoolSize = 4
        maxPoolSize = 8
        queueCapacity = 2000
        setThreadNamePrefix("watermark-")
        setWaitForTasksToCompleteOnShutdown(true)
        setAwaitTerminationSeconds(30)
        initialize()
    }
}
