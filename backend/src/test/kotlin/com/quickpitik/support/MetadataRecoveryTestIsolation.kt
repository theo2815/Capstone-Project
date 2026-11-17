package com.quickpitik.support

import org.springframework.beans.factory.config.BeanFactoryPostProcessor
import org.springframework.boot.test.context.TestConfiguration
import org.springframework.context.annotation.Bean
import org.springframework.scheduling.annotation.ScheduledAnnotationBeanPostProcessor
import org.springframework.scheduling.config.TaskManagementConfigUtils

/**
 * No @Scheduled method runs on a timer in a test context. Cached test contexts share
 * PostgreSQL but not their mocks, so an idle context's sweep would act on another
 * class's rows with the wrong mocks. Tests call the sweep methods directly.
 */
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
        override fun postProcessAfterInitialization(bean: Any, beanName: String): Any = bean
    }
}
