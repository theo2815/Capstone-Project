package com.quickpitik.support

import org.junit.jupiter.api.Tag
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.context.ActiveProfiles
import org.springframework.test.context.DynamicPropertyRegistry
import org.springframework.test.context.DynamicPropertySource
import org.testcontainers.containers.PostgreSQLContainer

/**
 * Base class for tests that need a real database.
 *
 * Everything else in this suite is a Mockito unit test, which is fast and needs
 * nothing installed — but it also means Flyway, the JPA mappings, and every
 * partial index in `db/migration` had never been executed by a test. This class
 * is what makes exercising them possible.
 *
 * **Tagged `integration`, so `./gradlew test` skips it.** Docker is required
 * only for `./gradlew integrationTest`; the unit suite's no-dependencies
 * contract (`backend/CLAUDE.md`) is unaffected.
 *
 * The container is a **singleton** — started once on first class-load and left
 * to Ryuk to reap at JVM exit, rather than `@Container` per class, which would
 * pay the ~3s startup for every test class. Schema state therefore carries
 * across classes: build your own fixtures with unique keys and never assume an
 * empty table. `BootstrapAdminRunner` and `BootstrapPhotosRunner` also run on
 * this database, so `users` and `photos` are non-empty from the first boot.
 */
@Tag("integration")
@SpringBootTest
@ActiveProfiles("test")
abstract class PostgresIntegrationTest {

    companion object {
        @JvmStatic
        private val postgres: PostgreSQLContainer<*> =
            PostgreSQLContainer("postgres:16-alpine").apply {
                withDatabaseName("quickpitik_test")
                withUsername("quickpitik")
                withPassword("quickpitik")
                // Reused across every class in the run; see the class docblock.
                start()
            }

        @JvmStatic
        @DynamicPropertySource
        fun datasourceProperties(registry: DynamicPropertyRegistry) {
            registry.add("spring.datasource.url", postgres::getJdbcUrl)
            registry.add("spring.datasource.username", postgres::getUsername)
            registry.add("spring.datasource.password", postgres::getPassword)
        }
    }
}
