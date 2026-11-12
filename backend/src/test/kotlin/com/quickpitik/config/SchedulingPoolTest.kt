package com.quickpitik.config

import org.junit.jupiter.api.Test
import org.springframework.boot.autoconfigure.task.TaskSchedulingProperties
import org.springframework.boot.context.properties.bind.Binder
import org.springframework.boot.env.YamlPropertySourceLoader
import org.springframework.core.env.StandardEnvironment
import org.springframework.core.io.ClassPathResource
import kotlin.test.assertEquals

// Boot silently ignores an unknown key: `pool-size` (instead of `pool.size`) left
// every @Scheduled job on one thread from 2026-09-14 to 2026-11-11.
class SchedulingPoolTest {

    @Test fun `application yml gives the scheduler six threads`() {
        val env = StandardEnvironment()
        YamlPropertySourceLoader().load("yml", ClassPathResource("application.yml")).forEach { env.propertySources.addLast(it) }
        val bound = Binder.get(env).bindOrCreate("spring.task.scheduling", TaskSchedulingProperties::class.java)
        assertEquals(6, bound.pool.size)
    }
}
