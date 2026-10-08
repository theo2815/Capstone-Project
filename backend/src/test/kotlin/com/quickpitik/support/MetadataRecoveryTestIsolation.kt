package com.quickpitik.support

import com.quickpitik.service.photographer.PhotoMetadataService
import org.springframework.beans.factory.config.BeanFactoryPostProcessor
import org.springframework.boot.test.context.TestConfiguration
import org.springframework.context.annotation.Bean
import org.springframework.scheduling.annotation.ScheduledAnnotationBeanPostProcessor
import org.springframework.scheduling.config.TaskManagementConfigUtils

/** Cached test contexts share PostgreSQL, but their storage/extractor mocks do not. */
@TestConfiguration(proxyBeanMethods = false)
class MetadataRecoveryTestIsolation {
    companion object {
        @Bean
        @JvmStatic
        fun isolateMetadataRecoveryTimers(): BeanFactoryPostProcessor = BeanFactoryPostProcessor { factory ->
            val definition = factory.getBeanDefinition(TaskManagementConfigUtils.SCHEDULED_ANNOTATION_PROCESSOR_BEAN_NAME)
            definition.factoryBeanName = null
            definition.factoryMethodName = null
            definition.beanClassName = TestSchedules::class.java.name
        }
    }

    class TestSchedules : ScheduledAnnotationBeanPostProcessor() {
        override fun postProcessAfterInitialization(bean: Any, beanName: String): Any =
            // Manual recoverDue calls retain the real recovery policy and assertions.
            if (bean is PhotoMetadataService || bean is com.quickpitik.service.photos.PhotoErasureSweep) bean
            else super.postProcessAfterInitialization(bean, beanName)
    }
}
