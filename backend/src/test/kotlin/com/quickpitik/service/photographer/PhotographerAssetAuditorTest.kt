package com.quickpitik.service.photographer

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.core.read.ListAppender
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.slf4j.LoggerFactory
import java.util.UUID
import kotlin.test.assertTrue

// A storage error (S3StorageService rethrows every non-404, e.g. a 403 after a
// token rotation) is not proof the object exists: the audit must say it could
// not check, never "all keys present".
class PhotographerAssetAuditorTest {

    @Test
    fun `a storage error is logged as unchecked, not counted present`() {
        val settings = Mockito.mock(PhotographerSettingsRepository::class.java)
        val selfies = Mockito.mock(UserSelfieRepository::class.java)
        val storage = Mockito.mock(StorageService::class.java)
        Mockito.`when`(settings.findAll())
            .thenReturn(listOf(PhotographerSettings(userId = UUID.randomUUID(), watermarkS3Key = "wm/key.png")))
        Mockito.`when`(selfies.findAll()).thenReturn(emptyList())
        Mockito.`when`(storage.exists("wm/key.png")).thenThrow(IllegalStateException("403 Forbidden"))

        val logger = LoggerFactory.getLogger(PhotographerAssetAuditor::class.java) as Logger
        val appender = ListAppender<ILoggingEvent>().apply { start() }
        logger.addAppender(appender)
        try {
            PhotographerAssetAuditor(settings, selfies, storage).audit()

            assertTrue(appender.list.none { it.level == Level.INFO && it.formattedMessage.contains("present") })
            assertTrue(
                appender.list.any {
                    it.level == Level.WARN && it.formattedMessage.contains("wm/key.png") &&
                        it.formattedMessage.contains("403 Forbidden")
                },
            )
        } finally {
            logger.detachAppender(appender)
            appender.stop()
        }
    }
}
